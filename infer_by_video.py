import os
import numpy as np
from argparse import ArgumentParser
import csv
import pandas as pd
import torch
import cv2
import time
import concurrent.futures
from concurrent.futures import wait, FIRST_COMPLETED

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.config import ConfigData, ConfigFeat, ConfigPatchCore, ConfigDraw
from utils.tictoc import tic, toc
from utils.metrics import calc_imagewise_metrics, calc_pixelwise_metrics, calc_roc_best_score
from utils.visualize import draw_roc_curve, draw_distance_graph, draw_heatmap, pickup_patch_from_coreset_patch

from datasets.mvtec_dataset import MVTecDatasetInfer, MVTecDataset

from models.feat_extract import FeatExtract
from models.patchcore import PatchCore


def arg_parser():
    parser = ArgumentParser()
    # environment related
    parser.add_argument('-n', '--num_cpu_max', default=4, type=int,
                        help='number of CPUs for parallel reading input images')
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    # I/O and visualization related
    parser.add_argument('-pp', '--path_parent', type=str, default='./mvtec_anomaly_detection',
                        help='parent path of data input path')
    parser.add_argument('-pr', '--path_result', type=str, default='./result',
                        help='output path of figure image as the evaluation result')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='save visualization of localization')
    parser.add_argument('-srf', '--size_receptive_field', type=int, default=15,
                        help='estimate and specify receptive field size (odd number)')
    parser.add_argument('-mv', '--mode_visualize', type=str, default='eval',
                        choices=['eval', 'infer'], help='set mode, [eval] or [infer]')

    parser.add_argument('--thr_save_dir', type=str, default='output', help='Specify the directory to output img thresholds')

    parser.add_argument('--score_max', type=float, help='Value for normalization to use when visualizing')

    parser.add_argument('--frame_skip', type=int, default=8, help='Set the number of frames to skip when visualizing video')

    # data loader related
    parser.add_argument('-bs', '--batch_size', type=int, default=16,
                        help='batch-size for feature extraction by ImageNet model')
    parser.add_argument('-sr', '--size_resize', nargs=2, type=int, default=[256, 256],
                        help='size of resizing input image')
    parser.add_argument('-sc', '--size_crop', nargs=2, type=int, default=[224, 224],
                        help='size of cropping after resize')
    parser.add_argument('-fh', '--flip_horz', action='store_true', help='flip horizontal')
    parser.add_argument('-fv', '--flip_vert', action='store_true', help='flip vertical')
    parser.add_argument('-tt', '--types_data', nargs='*', type=str, default=None)

    parser.add_argument('--path_input_video', type=str)
    parser.add_argument('--path_coreset_patch_img', type=str)

    # feature extraction related
    parser.add_argument('-b', '--backbone', type=str,
                        default='torchvision.models.wide_resnet50_2',
                        help='specify torchvision model with the full path')
    parser.add_argument('-w', '--weight', type=str,
                        default='torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1',
                        help='specify the trained weights of torchvision model with the full path')
    parser.add_argument('-lm', '--layer_map', nargs='+', type=str,
                        default=['layer2[-1]', 'layer3[-1]'],
                        help='specify layers to extract feature map')

    parser.add_argument('--merge_dst_layer', type=str, default='layer2[-1]',
                        help='layer, which specifies a layer with spatial information as a reference when merging layer')

    parser.add_argument('--faiss_save_dir', type=str, default='output', help='Specify the directory to output faiss index')
    parser.add_argument('--coreset_patch_save_dir', type=str, default='output', help='Specify where to save coreset patch')

    # patchification related
    parser.add_argument('-sp', '--size_patch', type=int, default=3,
                        help='patch pixel of feature map for increasing receptive field size')
    parser.add_argument('-de', '--dim_each_feat', type=int, default=1024,
                        help='dimension of extract feature (at 1st adaptive average pooling)')
    parser.add_argument('-dm', '--dim_merge_feat', type=int, default=1024,
                        help='dimension after layer feature merging (at 2nd adaptive average pooling)')
    # coreset related
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='specify a random-seed for k-center-greedy')
    parser.add_argument('-ns', '--num_split_seq', type=int, default=1,
                        help='percentage of coreset to all patch features')
    parser.add_argument('-pc', '--percentage_coreset', type=float, default=0.01,
                        help='percentage of coreset to all patch features')
    parser.add_argument('-ds', '--dim_sampling', type=int, default=128,
                        help='dimension to project features for sampling')
    parser.add_argument('-ni', '--num_initial_coreset', type=int, default=10,
                        help='number of samples to initially randomly select coreset')

    parser.add_argument('--faiss_index_path', type=str, help='Specify the file where the faiss index is saved')

    # Nearest-Neighbor related
    parser.add_argument('-k', '--k', type=int, default=5,
                        help='nearest neighbor\'s k for coreset searching')
    # post precessing related
    parser.add_argument('-pod', '--pixel_outer_decay', type=int, default=0,
                        help='number of outer pixels to decay anomaly score')

    args = parser.parse_args()
    return args


def apply_colormap_on_scoremap(scoremap, colormap=cv2.COLORMAP_JET):
    normalized_scoremap = cv2.normalize(scoremap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored_scoremap = cv2.applyColorMap(normalized_scoremap, colormap)
    return colored_scoremap


def blend_images(image, colored_scoremap, alpha=0.5):
    blended = cv2.addWeighted(image, alpha, colored_scoremap, 1 - alpha, 0)
    return blended


def apply_patchcore(args, feat_ext, patchcore, frame, cfg_draw):
    height, width, channel = frame.shape

    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
    type_name = 'infer_video'
    feat_test = {}
    feat_test[type_name] = feat_ext.extract(resized_frame[None], case='', show_progress=False)
    D, D_max, I = patchcore.localization(feat_test, show_progress=False)

    score_map = D[type_name][0]
    resized_scoremap = cv2.resize(score_map, (width, height), interpolation=cv2.INTER_LINEAR)
    colored_scoremap = apply_colormap_on_scoremap(resized_scoremap)
    blended_image = blend_images(frame, colored_scoremap, alpha=0.5)

    return resized_frame, blended_image, score_map, D, D_max, I


def inference(args, feat_ext, patchcore, cfg_draw):
    patchcore.reset_neighbor()
    patchcore.load_neighbor_from_file(args.faiss_index_path)

    cap = cv2.VideoCapture(args.path_input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0

    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    fig_width = 10 * max(1, cfg_draw.aspect_figure)
    fig_height = height = 18
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100, facecolor='white')
    plt.rcParams['font.size'] = 10

    # original image
    ax_image = plt.subplot2grid((7, 2), (0, 0), rowspan=1, colspan=1)
    im_image = ax_image.imshow(frame)

    # score map
    ax_score = plt.subplot2grid((7, 2), (0, 1), rowspan=1, colspan=1)
    resized_frame, processed_frame, score_map, distance, distance_max, index = apply_patchcore(
        args,
        feat_ext,
        patchcore,
        frame,
        cfg_draw)
    im_score = ax_score.imshow(score_map)
    # plt.colorbar(im_score, ax_score)

    # overlay heatmap
    ax_heatmap = plt.subplot2grid((42, 2), (7, 0), rowspan=10, colspan=1)
    im_heatmap = ax_heatmap.imshow(processed_frame)

    # image by normalize by score max
    ax_score_max = plt.subplot2grid((42, 2), (7, 1), rowspan=10, colspan=1)
    score_max = args.score_max if args.score_max else distance_max
    im_score_max = ax_score_max.imshow((resized_frame.astype(np.float32) * (score_map / score_max)[..., None]).astype(np.uint8))

    # image patch
    ax_patch = plt.subplot2grid((21, 1), (10, 0), rowspan=11, colspan=1)
    coreset_patch_img = patchcore.load_coreset_patch_from_file(args.path_coreset_patch_img)

    idx_patch_one_img = index['infer_video'][0, :, 0]
    imgs_shape = resized_frame.shape
    HW_map = feat_ext.HW_map()
    img_patch = pickup_patch_from_coreset_patch(
        idx_patch_one_img,
        coreset_patch_img,
        imgs_shape,
        HW_map,
        cfg_draw.size_receptive_field)

    im_patch = ax_patch.imshow(img_patch, interpolation='none')

    def update(frame):
        nonlocal args
        nonlocal frame_count
        nonlocal im_image, ax_image
        nonlocal im_score, ax_score
        nonlocal im_score_max, ax_score_max
        nonlocal im_heatmap, ax_heatmap
        nonlocal im_patch, ax_patch
        nonlocal coreset_patch_img

        while frame_count < args.frame_skip:
            if not cap.grab():
                return im_image, im_score, im_score_max, im_heatmap, im_patch
            frame_count += 1

        if not cap.grab():
            return im_image, im_score, im_score_max, im_heatmap, im_patch

        ret, frame = cap.retrieve()
        frame_count = 0

        if ret:
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # original image
            im_image.set_data(frame)

            # score map
            resized_frame, processed_frame, score_map, distance, distance_max, index = apply_patchcore(
                args,
                feat_ext,
                patchcore,
                frame,
                cfg_draw)

            im_score.set_data(score_map)
            # plt.colorbar(im_score, ax_score)

            # overlay heatmap
            im_heatmap.set_data(processed_frame)

            # image by normalize by score max
            im_score_max.set_data((resized_frame.astype(np.float32) * (score_map / score_max)[..., None]).astype(np.uint8))

            # image patch
            idx_patch_one_img = index['infer_video'][0, :, 0]
            imgs_shape = resized_frame.shape
            HW_map = feat_ext.HW_map()
            img_patch = pickup_patch_from_coreset_patch(
                idx_patch_one_img,
                coreset_patch_img,
                imgs_shape,
                HW_map,
                cfg_draw.size_receptive_field)

            im_patch.set_data(img_patch)

        return im_image, im_score, im_score_max, im_heatmap, im_patch

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = total_frames // args.frame_skip
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)

    # 動画ファイルとして保存
    output_video_path = 'output.mp4'
    ani.save(output_video_path, writer='ffmpeg')
    cap.release()


def inference__(args, feat_ext, patchcore, cfg_draw):
    patchcore.reset_neighbor()
    patchcore.load_neighbor_from_file(args.faiss_index_path)

    cap = cv2.VideoCapture(args.path_input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_filename = 'output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    fig_width = 10 * max(1, cfg_draw.aspect_figure)

    fig_height = height = 18
    plt.ion()
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100, facecolor='white')
    plt.rcParams['font.size'] = 10
    ax = plt.subplot2grid((7, 2), (0, 0), rowspan=1, colspan=1)

    while cap.isOpened():
        start_time = time.time()

        ret = cap.grab()
        frame_count += 1

        if not ret:
            break

        if frame_count % args.frame_skip != 0:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            break

        processed_frame = apply_patchcore(args, feat_ext, patchcore, frame, cfg_draw)

        ax.imshow(processed_frame)
        plt.draw()
        plt.pause(0.1)
        ax.clear()
        # cv2.imshow('Processed Frame', processed_frame)

        # out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # time_to_wait = max(1.0 / fps - (time.time() - start_time), 0)
        time_to_wait = 0.005
        time.sleep(time_to_wait)

    # 動画ファイルとして保存
    output_video_path = 'output.mp4'
    writer = animation.FFMpegWriter(fps=20, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])

    cap.release()
    # out.release()
    cv2.destroyAllWindows()



def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ConfigData(args)  # static define for speed-up
    cfg_feat = ConfigFeat(args)
    cfg_patchcore = ConfigPatchCore(args)
    cfg_draw = ConfigDraw(args)

    feat_ext = FeatExtract(cfg_feat)
    patchcore = PatchCore(cfg_patchcore, feat_ext.HW_map())

    inference(args, feat_ext, patchcore, cfg_draw)


if __name__ == '__main__':
    args = arg_parser()
    main(args)