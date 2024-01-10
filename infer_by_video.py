import os
import numpy as np
from argparse import ArgumentParser
import torch
import cv2
from PIL import Image
from tqdm import tqdm

from utils.config import ConfigData, ConfigFeat, ConfigPatchCore, ConfigDraw
from utils.tictoc import tic, toc
from utils.metrics import calc_imagewise_metrics, calc_pixelwise_metrics, calc_roc_best_score
from utils.visualize import draw_roc_curve, draw_distance_graph, draw_heatmap, pickup_patch_from_coreset_patch, draw_heatmap

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

    parser.add_argument('--frame_skip', type=int, default=4, help='Set the number of frames to skip when visualizing video')

    parser.add_argument('--display_video', action='store_true', help='If you want to display video during inference, set')

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


def apply_patchcore(args, feat_ext, patchcore, frame, cfg_draw):
    img = cv2.resize(frame, (ConfigData.SHAPE_MIDDLE[1], ConfigData.SHAPE_MIDDLE[0]), interpolation=cv2.INTER_AREA)

    img = img[
          ConfigData.pixel_cut[0]:(ConfigData.SHAPE_INPUT[0] + ConfigData.pixel_cut[0]),
          ConfigData.pixel_cut[1]:(ConfigData.SHAPE_INPUT[1] + ConfigData.pixel_cut[1])
    ]

    type_name = 'infer_video'
    feat_test = {}
    feat_test[type_name] = feat_ext.extract(img[None], case='', show_progress=False)
    D, D_max, I = patchcore.localization(feat_test, show_progress=False)

    return img, D, D_max, I


def inference(args, feat_ext, patchcore, cfg_draw):
    patchcore.reset_neighbor()
    patchcore.load_neighbor_from_file(args.faiss_index_path)

    cap = cv2.VideoCapture(args.path_input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pbar = tqdm(total=total_frames, desc='Processing Video')

    frame_width = 1000
    frame_height = 1800

    output_filename = 'output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    coreset_patch_img = patchcore.load_coreset_patch_from_file(args.path_coreset_patch_img)

    frame_count = 0
    while cap.isOpened():
        ret = cap.grab()
        frame_count += 1

        pbar.update(1)

        if not ret:
            break

        if frame_count % args.frame_skip != 0:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            break

        resized_frame, distance, distance_max, index = apply_patchcore(
            args,
            feat_ext,
            patchcore,
            frame,
            cfg_draw)

        type_data = 'infer_video'
        imgs_test = {type_data: resized_frame[None]}
        y = {}
        files_test = {type_data: [args.path_input_video]}

        img_figure_dict = draw_heatmap(
            type_data,
            cfg_draw,
            distance,
            y,
            distance_max,
            imgs_test,
            files_test,
            coreset_patch_img,
            index,
            None,
            feat_ext.HW_map(),
            coreset_patch_img=coreset_patch_img,
            is_save_file=False,
            is_tqdm=False
        )

        img_figure = img_figure_dict['infer_video'][:, :, ::-1]

        if args.display_video:
            cv2.imshow('Processed Frame', img_figure)

        out.write(img_figure)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pbar.close()
    cap.release()
    out.release()
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
