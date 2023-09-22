import os
import numpy as np
from argparse import ArgumentParser

from utils.config import ConfigData, ConfigFeat, ConfigPatchCore, ConfigDraw
from utils.tictoc import tic, toc
from utils.metrics import calc_imagewise_metrics, calc_pixelwise_metrics
from utils.visualize import draw_roc_curve, draw_distance_graph, draw_heatmap
from datasets.mvtec_dataset import MVTecDataset
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
    # data loader related
    parser.add_argument('-bs', '--batch_size', type=int, default=16,
                        help='batch-size for feature extraction by ImageNet model')
    parser.add_argument('-sr', '--size_resize', type=int, default=256,
                        help='size of resizing input image')
    parser.add_argument('-sc', '--size_crop', type=int, default=224,
                        help='size of cropping after resize')
    # feature extraction related
    parser.add_argument('-b', '--backbone', type=str,
                        default='torchvision.models.wide_resnet50_2',
                        help='specify torchvision model with the full path')
    parser.add_argument('-w', '--weight', type=str,
                        default='torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1',
                        help='specify the trained weights of torchvision model with the full path.')
    parser.add_argument('-lm', '--layer_map', nargs='+', type=str,
                        default=['layer2[-1]', 'layer3[-1]'],
                        help='specify layers to extract feature map')
    # patchification related
    parser.add_argument('-sp', '--size_patch', type=int, default=3,
                        help='patch pixel of feature map for increasing receptive field size')
    parser.add_argument('-de', '--dim_each_feat', type=int, default=1024,
                        help='dimension of extract feature (at 1st adaptive average pooling)')
    parser.add_argument('-dm', '--dim_merge_feat', type=int, default=1024,
                        help='dimension after layer feature merging (at 2nd adaptive average pooling)')
    # coreset related
    parser.add_argument('-s', '--seed', type=float, default=0,
                        help='specify a random-seed for k-center-greedy')
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


def apply_patchcore(type_data, feat_ext, patchcore, cfg_draw):
    print('\n----> PatchCore processing in %s start' % type_data)
    tic()

    # read images
    MVTecDataset(type_data)

    # extract features
    feat_train = feat_ext.extract(MVTecDataset.imgs_train, is_train=True)
    feat_test = feat_ext.extract(MVTecDataset.imgs_test, is_train=False)

    # coreset-reduced patch-feature memory bank
    idx_coreset = patchcore.compute_greedy_coreset_idx(feat_train)
    feat_train = feat_train[idx_coreset]
    patchcore.set_nearest_neighbor(feat_train)

    # Sub-Image Anomaly Detection with Deep Pyramid Correspondences
    D, D_max, I = patchcore.localization(feat_test)

    # measure per image
    fpr_img, tpr_img, rocauc_img = calc_imagewise_metrics(D)
    print('%s imagewise ROCAUC: %.3f' % (type_data, rocauc_img))
    fpr_pix, tpr_pix, rocauc_pix = calc_pixelwise_metrics(D, MVTecDataset.gts_test)
    print('%s pixelwise ROCAUC: %.3f' % (type_data, rocauc_pix))

    toc(tag=('----> PatchCore processing in %s end, elapsed time' % type_data))

    draw_distance_graph(type_data, cfg_draw, D)
    if args.verbose:
        draw_heatmap(type_data, cfg_draw, D, MVTecDataset.gts_test, D_max,
                     MVTecDataset.imgs_test, MVTecDataset.files_test,
                     idx_coreset, I, MVTecDataset.imgs_train, feat_ext.HW_map())

    return [fpr_img, tpr_img, rocauc_img, fpr_pix, tpr_pix, rocauc_pix]


def main(args):
    ConfigData(args)  # static define to speed-up
    cfg_feat = ConfigFeat(args)
    cfg_patchcore = ConfigPatchCore(args)
    cfg_draw = ConfigDraw(args)

    feat_ext = FeatExtract(cfg_feat)
    patchcore = PatchCore(cfg_patchcore, feat_ext.HW_map())

    os.makedirs(args.path_result, exist_ok=True)
    for type_data in ConfigData.types_data:
        os.makedirs(os.path.join(args.path_result, type_data), exist_ok=True)

    fpr_img = {}
    tpr_img = {}
    rocauc_img = {}
    fpr_pix = {}
    tpr_pix = {}
    rocauc_pix = {}

    # loop for types of data
    for type_data in ConfigData.types_data:
        result = apply_patchcore(type_data, feat_ext, patchcore, cfg_draw)

        fpr_img[type_data] = result[0]
        tpr_img[type_data] = result[1]
        rocauc_img[type_data] = result[2]

        fpr_pix[type_data] = result[3]
        tpr_pix[type_data] = result[4]
        rocauc_pix[type_data] = result[5]

    draw_roc_curve(cfg_draw, fpr_img, tpr_img, rocauc_img, fpr_pix, tpr_pix, rocauc_pix)

    rocauc_img_ = np.array([rocauc_img[type_data] for type_data in ConfigData.types_data])
    rocauc_pix_ = np.array([rocauc_pix[type_data] for type_data in ConfigData.types_data])
    for type_data in ConfigData.types_data:
        print('rocauc_img[%s] = %.3f' % (type_data, rocauc_img[type_data]))
    print('np.mean(rocauc_img_) = %.3f' % np.mean(rocauc_img_))
    for type_data in ConfigData.types_data:
        print('rocauc_pix[%s] = %.3f' % (type_data, rocauc_pix[type_data]))
    print('np.mean(rocauc_pix_) = %.3f' % np.mean(rocauc_pix_))


if __name__ == '__main__':
    args = arg_parser()
    main(args)
