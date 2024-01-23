import os
from argparse import ArgumentParser
import json

import numpy as np
import torch
import multiprocessing as mp

from utils.config import ConfigData, ConfigFeat, ConfigPatchCore, ConfigDraw
from utils.tictoc import tic, toc
from utils.metrics import calc_imagewise_metrics, calc_pixelwise_metrics
from utils.visualize import draw_curve, draw_distance_graph, draw_heatmap
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
    parser.add_argument('-pd', '--path_data', type=str, default='./mvtec_anomaly_detection',
                        help='parent path of input data path')
    parser.add_argument('-pt', '--path_trained', type=str, default='./trained',
                        help='output path of trained products')
    parser.add_argument('-pr', '--path_result', type=str, default='./result',
                        help='output path of figure image as the evaluation result')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='save visualization of localization')
    parser.add_argument('-srf', '--size_receptive', type=int, default=9,
                        help='estimate and specify receptive field size (odd number)')
    parser.add_argument('-mv', '--mode_visualize', type=str, default='eval',
                        choices=['eval', 'infer'], help='set mode, [eval] or [infer]')
    parser.add_argument('-sm', '--score_max', type=float, default=None,
                        help='value for normalization of visualizing')

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
    parser.add_argument('-lw', '--layer_weight', nargs='+', type=float,
                        default=[1.0, 1.0],
                        help='specify layers weights for merge of feature map')
    parser.add_argument('-lmr', '--layer_merge_ref', type=str, default='layer2[-1]',
                        help='specify the layer to use as a reference for spatial size when merging feature maps')

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

    print('args =\n', args)
    return args


def check_args(args):
    assert 0 < args.num_cpu_max < os.cpu_count()
    assert os.path.isdir(args.path_data)
    assert (args.size_receptive % 2) == 1
    assert args.size_receptive > 0
    if args.score_max is not None:
        assert args.score_max > 0
    assert args.batch_size > 0
    assert args.size_resize[0] > 0
    assert args.size_resize[1] > 0
    assert args.size_crop[0] > 0
    assert args.size_crop[1] > 0
    if args.types_data is not None:
        for type_data in args.types_data:
            assert os.path.isdir('%s/%s' % (args.path_data, type_data))
    assert len(args.layer_map) == len(args.layer_weight)
    assert args.layer_merge_ref in args.layer_map
    assert args.size_patch > 0
    assert args.dim_each_feat > 0
    assert args.dim_merge_feat > 0
    assert args.num_split_seq > 0
    assert 0.0 < args.percentage_coreset <= 1.0
    assert args.dim_sampling > 0
    assert args.num_initial_coreset > 0
    assert args.k > 0
    assert args.pixel_outer_decay >= 0


def set_seed(seed, gpu=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def apply_patchcore(type_data, feat_ext, patchcore, cfg_draw):
    print('\n----> PatchCore processing in %s start' % type_data)
    tic()

    # read images
    MVTecDataset(type_data)

    # reset neighbor
    patchcore.reset_faiss_index()

    # reset total index
    idx_coreset_total = []

    # loop of split-sequential to apply k-center-greedy
    num_pitch = int(np.ceil(len(MVTecDataset.imgs_train) / patchcore.num_split_seq))
    for i_split in range(patchcore.num_split_seq):
        # extract features
        i_from = i_split * num_pitch
        i_to = min(((i_split + 1) * num_pitch), len(MVTecDataset.imgs_train))
        if patchcore.num_split_seq > 1:
            print('[split%02d] image index range is %d~%d' % (i_split, i_from, (i_to - 1)))
        feat_train = feat_ext.extract(MVTecDataset.imgs_train[i_from:i_to])

        # coreset-reduced patch-feature memory bank
        idx_coreset = patchcore.compute_greedy_coreset_idx(feat_train)
        feat_train = feat_train[idx_coreset]

        # add feature as neighbor
        patchcore.add_neighbor(feat_train)

        # stock index
        offset_split = i_from * feat_ext.HW_map()[0] * feat_ext.HW_map()[1]
        idx_coreset_total.append(idx_coreset + offset_split)

    # save faiss index    
    patchcore.save_faiss_index(type_data)

    # concat index
    idx_coreset_total = np.hstack(idx_coreset_total)

    # save and get images of coreset
    imgs_coreset = patchcore.save_coreset(idx_coreset_total, type_data,
                                          MVTecDataset.imgs_train, feat_ext.HW_map(),
                                          cfg_draw.size_receptive)

    # extract features
    feat_test = {}
    for type_test in MVTecDataset.imgs_test.keys():
        feat_test[type_test] = feat_ext.extract(MVTecDataset.imgs_test[type_test],
                                                case='test (case:%s)' % type_test)

    # Sub-Image Anomaly Detection with Deep Pyramid Correspondences
    D, D_max, I = patchcore.localization(feat_test, feat_ext.HW_map())

    # measure per image
    fpr_img, tpr_img, rocauc_img, pre_img, rec_img, prauc_img = calc_imagewise_metrics(D)
    print('%s imagewise ROCAUC: %.3f' % (type_data, rocauc_img))

    # measure per pixel
    (fpr_pix, tpr_pix, rocauc_pix,
     pre_pix, rec_pix, prauc_pix, thresh_opt) = calc_pixelwise_metrics(D, MVTecDataset.gts_test)
    print('%s pixelwise ROCAUC: %.3f' % (type_data, rocauc_pix))

    # save optimal threshold
    np.savetxt('%s/%s_thr.txt' % (args.path_trained, type_data),
               np.array([thresh_opt]), fmt='%.3f')

    toc(tag=('----> PatchCore processing in %s end, elapsed time' % type_data))

    draw_distance_graph(type_data, cfg_draw, D, rocauc_img)
    if cfg_draw.verbose:
        draw_heatmap(type_data, cfg_draw, D, MVTecDataset.gts_test, D_max, I,
                     MVTecDataset.imgs_test, MVTecDataset.files_test,
                     imgs_coreset, feat_ext.HW_map())

    return [fpr_img, tpr_img, rocauc_img, pre_img, rec_img, prauc_img,
            fpr_pix, tpr_pix, rocauc_pix, pre_pix, rec_pix, prauc_pix]


def main(args):
    ConfigData(args)  # static define for speed-up
    cfg_feat = ConfigFeat(args)
    cfg_patchcore = ConfigPatchCore(args)
    cfg_draw = ConfigDraw(args)

    feat_ext = FeatExtract(cfg_feat)
    patchcore = PatchCore(cfg_patchcore)

    os.makedirs(args.path_result, exist_ok=True)
    for type_data in ConfigData.types_data:
        os.makedirs('%s/%s' % (args.path_result, type_data), exist_ok=True)

    fpr_img = {}
    tpr_img = {}
    rocauc_img = {}
    pre_img = {}
    rec_img = {}
    prauc_img = {}
    fpr_pix = {}
    tpr_pix = {}
    rocauc_pix = {}
    pre_pix = {}
    rec_pix = {}
    prauc_pix = {}

    # loop for types of data
    for type_data in ConfigData.types_data:
        set_seed(seed=args.seed, gpu=(not args.cpu))

        result = apply_patchcore(type_data, feat_ext, patchcore, cfg_draw)

        fpr_img[type_data] = result[0]
        tpr_img[type_data] = result[1]
        rocauc_img[type_data] = result[2]
        pre_img[type_data] = result[3]
        rec_img[type_data] = result[4]
        prauc_img[type_data] = result[5]

        fpr_pix[type_data] = result[6]
        tpr_pix[type_data] = result[7]
        rocauc_pix[type_data] = result[8]
        pre_pix[type_data] = result[9]
        rec_pix[type_data] = result[10]
        prauc_pix[type_data] = result[11]

    rocauc_img_mean = np.array([rocauc_img[type_data] for type_data in ConfigData.types_data])
    rocauc_img_mean = np.mean(rocauc_img_mean)
    prauc_img_mean = np.array([prauc_img[type_data] for type_data in ConfigData.types_data])
    prauc_img_mean = np.mean(prauc_img_mean)
    rocauc_pix_mean = np.array([rocauc_pix[type_data] for type_data in ConfigData.types_data])
    rocauc_pix_mean = np.mean(rocauc_pix_mean)
    prauc_pix_mean = np.array([prauc_pix[type_data] for type_data in ConfigData.types_data])
    prauc_pix_mean = np.mean(prauc_pix_mean)

    draw_curve(cfg_draw, fpr_img, tpr_img, rocauc_img, rocauc_img_mean,
                         fpr_pix, tpr_pix, rocauc_pix, rocauc_pix_mean)
    draw_curve(cfg_draw, rec_img, pre_img, prauc_img, prauc_img_mean,
                         rec_pix, pre_pix, prauc_pix, prauc_pix_mean, False)

    for type_data in ConfigData.types_data:
        print('rocauc_img[%s] = %.3f' % (type_data, rocauc_img[type_data]))
    print('rocauc_img[mean] = %.3f' % rocauc_img_mean)
    for type_data in ConfigData.types_data:
        print('prauc_img[%s] = %.3f' % (type_data, prauc_img[type_data]))
    print('prauc_img[mean] = %.3f' % prauc_img_mean)
    for type_data in ConfigData.types_data:
        print('rocauc_pix[%s] = %.3f' % (type_data, rocauc_pix[type_data]))
    print('rocauc_pix[mean] = %.3f' % rocauc_pix_mean)
    for type_data in ConfigData.types_data:
        print('prauc_pix[%s] = %.3f' % (type_data, prauc_pix[type_data]))
    print('prauc_pix[mean] = %.3f' % prauc_pix_mean)

if __name__ == '__main__':
    args = arg_parser()
    check_args(args)
    main(args)

    with open('%s/args.json' % args.path_trained, mode='w') as f:
        json.dump(args.__dict__, f, indent=4)

