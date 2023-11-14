import os
import numpy as np
from argparse import ArgumentParser
import csv
import pandas as pd

from utils.config import ConfigData, ConfigFeat, ConfigPatchCore, ConfigDraw
from utils.tictoc import tic, toc
from utils.metrics import calc_imagewise_metrics, calc_pixelwise_metrics, calc_roc_best_score
from utils.visualize import draw_roc_curve, draw_distance_graph, draw_heatmap
from datasets.mvtec_dataset import MVTecDatasetInfer
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
    # Nearest-Neighbor related
    parser.add_argument('-k', '--k', type=int, default=5,
                        help='nearest neighbor\'s k for coreset searching')
    # post precessing related
    parser.add_argument('-pod', '--pixel_outer_decay', type=int, default=0,
                        help='number of outer pixels to decay anomaly score')

    args = parser.parse_args()
    return args


def read_best_thr(args, type_data):
    csv_file_path = os.path.join(args.thr_save_dir, f'{type_data}_thr.csv')
    df_csv = pd.read_csv(csv_file_path)
    thr = df_csv.loc[0, 'thr']

    return thr


def inference(score_list, file_list, threshold):
    data = []
    for type_test in score_list.keys():
        for i in range(len(score_list[type_test])):
            max_score = np.max(score_list[type_test][i])
            abnormal = int(threshold <= max_score)

            data.append({
                'path': file_list[type_test][i],
                'score': max_score,
                'abnormal': abnormal,
                'threshold': threshold
            })

    df = pd.DataFrame(data)

    return df


def apply_patchcore(args, type_data, feat_ext, patchcore, cfg_draw):
    print('\n----> PatchCore processing in %s start' % type_data)
    tic()

    # read images
    MVTecDatasetInfer(type_data)

    # reset neighbor
    patchcore.reset_neighbor()
    patchcore.load_neighbor(type_data)

    # extract features
    feat_test = {}
    for type_test in MVTecDatasetInfer.imgs_test.keys():
        feat_test[type_test] = feat_ext.extract(MVTecDatasetInfer.imgs_test[type_test], case='test (case:%s)' % type_test)

    # Sub-Image Anomaly Detection with Deep Pyramid Correspondences
    D, D_max, I = patchcore.localization(feat_test)
    if args.verbose:
        draw_heatmap(
            type_data,
            cfg_draw,
            D,
            MVTecDatasetInfer.gts_test,
            D_max,
            MVTecDatasetInfer.imgs_test,
            MVTecDatasetInfer.files_test,
            None,
            I,
            None,
            feat_ext.HW_map()
        )

    img_thr = read_best_thr(args, type_data)

    df_result = inference(D, MVTecDatasetInfer.files_test, img_thr)
    save_path = os.path.join(args.path_result, type_data, f'{type_data}_result.csv')
    df_result.to_csv(save_path)


def main(args):
    ConfigData(args)  # static define for speed-up
    cfg_feat = ConfigFeat(args)
    cfg_patchcore = ConfigPatchCore(args)
    cfg_draw = ConfigDraw(args)

    feat_ext = FeatExtract(cfg_feat)
    patchcore = PatchCore(cfg_patchcore, feat_ext.HW_map())

    os.makedirs(args.path_result, exist_ok=True)
    for type_data in ConfigData.types_data:
        os.makedirs(os.path.join(args.path_result, type_data), exist_ok=True)

    # loop for types of data
    for type_data in ConfigData.types_data:
        apply_patchcore(args, type_data, feat_ext, patchcore, cfg_draw)


if __name__ == '__main__':
    args = arg_parser()
    main(args)
