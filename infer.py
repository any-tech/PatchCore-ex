import os
import numpy as np
from argparse import ArgumentParser
import json

from utils.config import ConfigData, ConfigFeat, ConfigPatchCore, ConfigDraw
from utils.tictoc import tic, toc
from utils.visualize import draw_heatmap

from datasets.mvtec_dataset import MVTecDatasetOnlyTest

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
    parser.add_argument('-pv', '--path_video', type=str, default=None,
                        help='path of input video path (.mp4 or .avi or .mov)')
    parser.add_argument('-pt', '--path_trained', type=str, default='./trained',
                        help='output path of trained products')
    parser.add_argument('-pr', '--path_result', type=str, default='./result',
                        help='output path of figure image as the evaluation result')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='save visualization of localization')
    parser.add_argument('-sm', '--score_max', type=float, default=None,
                        help='value for normalization of visualizing')

    # data loader related
    parser.add_argument('-bs', '--batch_size', type=int, default=16,
                        help='batch-size for feature extraction by ImageNet model')
    parser.add_argument('-tt', '--types_data', nargs='*', type=str, default=None)

    # Nearest-Neighbor related
    parser.add_argument('-k', '--k', type=int, default=5,
                        help='nearest neighbor\'s k for coreset searching')
    # post precessing related
    parser.add_argument('-pod', '--pixel_outer_decay', type=int, default=0,
                        help='number of outer pixels to decay anomaly score')

    args = parser.parse_args()

    # adjust...
    if args.path_video is not None:
        args.path_data = None

    print('args =\n', args)
    return args


def check_args(args):
    assert 0 < args.num_cpu_max < os.cpu_count()
    if args.path_video is None:
        assert args.path_data is not None
        assert os.path.isdir(args.path_data)
    else:
        assert os.path.isfile(args.path_video)
        assert ((args.path_video.split('.')[-1].lower() == 'mp4') |
                (args.path_video.split('.')[-1].lower() == 'avi') |
                (args.path_video.split('.')[-1].lower() == 'mov'))
        assert len(args.types_data) == 1
    if args.score_max is not None:
        assert args.score_max > 0
    assert args.batch_size > 0
    if args.types_data is not None:
        if args.path_video is None:
            for type_data in args.types_data:
                assert os.path.exists('%s/%s' % (args.path_data, type_data))
    assert args.k > 0
    assert args.pixel_outer_decay >= 0


def summary_result(type_data, D, files_test, thresh):
    result = ['data-type filename anomaly-score threshold abnormal-judgement']
    for type_test in D.keys():
        for i_file in range(len(D[type_test])):
            D_max = np.max(D[type_test][i_file])
            flg_abnormal = int(thresh <= D_max)

            result.append('%s %s %.3f %.3f %d' %
                          (type_data, files_test[type_test][i_file],
                           D_max, thresh, flg_abnormal))

    filename_txt = '%s/%s/%s_result.txt' % (args.path_result, type_data, type_data)
    np.savetxt(filename_txt, result, fmt='%s')


def apply_patchcore_inference(args, type_data, feat_ext, patchcore, cfg_draw):
    print('\n----> inference-only PatchCore processing in %s start' % type_data)
    tic()

    # read images
    MVTecDatasetOnlyTest(type_data)

    # load neighbor
    patchcore.reset_faiss_index()
    patchcore.load_faiss_index(type_data)

    # extract features
    feat_test = {}
    for type_test in MVTecDatasetOnlyTest.imgs_test.keys():
        feat_test[type_test] = feat_ext.extract(MVTecDatasetOnlyTest.imgs_test[type_test],
                                                case='test (case:%s)' % type_test)

    # Sub-Image Anomaly Detection with Deep Pyramid Correspondences
    D, D_max, I = patchcore.localization(feat_test, feat_ext.HW_map())

    toc(tag=('----> inference-only PatchCore processing in %s end, elapsed time' % type_data))

    if args.verbose:
        imgs_coreset = patchcore.load_coreset(type_data)

        draw_heatmap(type_data, cfg_draw, D, None, D_max, I,
                     MVTecDatasetOnlyTest.imgs_test, MVTecDatasetOnlyTest.files_test,
                     imgs_coreset, feat_ext.HW_map())

    # load optimal threshold
    thresh = np.loadtxt('%s/%s_thr.txt' % (args.path_trained, type_data))

    # summary test result
    summary_result(type_data, D, MVTecDatasetOnlyTest.files_test, thresh)


def main(args):
    ConfigData(args, mode_train=False)  # static define for speed-up
    cfg_feat = ConfigFeat(args, mode_train=False)
    cfg_patchcore = ConfigPatchCore(args, mode_train=False)
    cfg_draw = ConfigDraw(args, mode_train=False)

    with open('%s/args.json' % args.path_trained, mode='r') as f:
        args_trained = json.load(f)
    ConfigData.follow(args_trained)
    cfg_feat.follow(args_trained)
    cfg_patchcore.follow(args_trained)
    cfg_draw.follow(args_trained)

    feat_ext = FeatExtract(cfg_feat)
    patchcore = PatchCore(cfg_patchcore)

    os.makedirs(args.path_result, exist_ok=True)
    for type_data in ConfigData.types_data:
        os.makedirs('%s/%s' % (args.path_result, type_data), exist_ok=True)

    # loop for types of data
    for type_data in ConfigData.types_data:
        apply_patchcore_inference(args, type_data, feat_ext, patchcore, cfg_draw)


if __name__ == '__main__':
    args = arg_parser()
    main(args)
