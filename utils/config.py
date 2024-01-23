import os
import numpy as np
import cv2
import torch

# https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html
MEAN = torch.FloatTensor([[[0.485, 0.456, 0.406]]])
STD = torch.FloatTensor([[[0.229, 0.224, 0.225]]])


class ConfigData:
    @classmethod
    def __init__(cls, args, mode_train=True):
        # file reading related
        if mode_train:
            cls.path_data = args.path_data
        else:
            cls.path_data = args.path_data
            cls.path_video = args.path_video
        cls.num_cpu_max = args.num_cpu_max
        if mode_train:
            cls.shuffle = (args.num_split_seq > 1)  # for k-center-greedy split
        else:
            cls.shuffle = None

        if mode_train:
            # input format related
            cls.SHAPE_MIDDLE = (args.size_resize[0], args.size_resize[1])  # (H, W)
            cls.SHAPE_INPUT = (args.size_crop[0], args.size_crop[1])  # (H, W)
            cls.pixel_cut = (int((cls.SHAPE_MIDDLE[0] - cls.SHAPE_INPUT[0]) / 2),
                             int((cls.SHAPE_MIDDLE[1] - cls.SHAPE_INPUT[1]) / 2))  # (H, W)
        else:
            cls.SHAPE_MIDDLE = None
            cls.SHAPE_INPUT = None
            cls.pixel_cut = None

        if mode_train:
            # augmantation related
            cls.flip_horz = args.flip_horz
            cls.flip_vert = args.flip_vert
        else:
            cls.flip_horz = None
            cls.flip_vert = None

        # collect types of data
        if args.types_data is None:
            types_data = [d for d in os.listdir(args.path_data)
                          if os.path.isdir('%s/%s' % (args.path_data, d))]
            cls.types_data = np.sort(np.array(types_data))
        else:
            cls.types_data = np.sort(np.array(args.types_data))

    @classmethod
    def follow(cls, args_trained):
        # input format related
        cls.SHAPE_MIDDLE = (args_trained['size_resize'][0],
                            args_trained['size_resize'][1])  # (H, W)
        cls.SHAPE_INPUT = (args_trained['size_crop'][0],
                           args_trained['size_crop'][1])  # (H, W)
        cls.pixel_cut = (int((cls.SHAPE_MIDDLE[0] - cls.SHAPE_INPUT[0]) / 2),
                         int((cls.SHAPE_MIDDLE[1] - cls.SHAPE_INPUT[1]) / 2))  # (H, W)


class ConfigFeat:
    def __init__(self, args, mode_train=True):
        # adjsut to environment
        if args.cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

        # batch-size for feature extraction by ImageNet model
        self.batch_size = args.batch_size

        if mode_train:
            # input format related
            self.SHAPE_INPUT = (args.size_crop[0], args.size_crop[1])  # (H, W)

            # base network
            self.backbone = args.backbone
            self.weight = args.weight

            # layer specification
            self.layer_map = args.layer_map
            self.layer_weight = args.layer_weight
            self.layer_merge_ref = args.layer_merge_ref

            # patch pixel of feature map for increasing receptive field size
            self.size_patch = args.size_patch

            # dimension of each layer feature (at 1st adaptive average pooling)
            self.dim_each_feat = args.dim_each_feat
            # dimension after layer feature merging (at 2nd adaptive average pooling)
            self.dim_merge_feat = args.dim_merge_feat
        else:
            self.SHAPE_INPUT = None
            self.backbone = None
            self.weight = None
            self.layer_map = None
            self.layer_weight = None
            self.layer_merge_ref = None
            self.size_patch = None
            self.dim_each_feat = None
            self.dim_merge_feat = None

        # adjust to the network's learning policy and the data conditions
        self.MEAN = MEAN.to(self.device)
        self.STD = STD.to(self.device)

    def follow(self, args_trained):
        # input format related
        self.SHAPE_INPUT = (args_trained['size_crop'][0],
                            args_trained['size_crop'][1])  # (H, W)
        # base network
        self.backbone = args_trained['backbone']
        self.weight = args_trained['weight']
        # layer specification
        self.layer_map = args_trained['layer_map']
        self.layer_weight = args_trained['layer_weight']
        self.layer_merge_ref = args_trained['layer_merge_ref']
        # patch pixel of feature map for increasing receptive field size
        self.size_patch = args_trained['size_patch']
        # dimension of each layer feature (at 1st adaptive average pooling)
        self.dim_each_feat = args_trained['dim_each_feat']
        # dimension after layer feature merging (at 2nd adaptive average pooling)
        self.dim_merge_feat = args_trained['dim_merge_feat']


class ConfigPatchCore:
    def __init__(self, args, mode_train=True):
        # adjsut to environment
        if args.cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

        if mode_train:
            # dimension after layer feature merging (at 2nd adaptive average pooling)
            self.dim_coreset_feat = args.dim_merge_feat

            # number split-sequential to apply k-center-greedy
            self.num_split_seq = args.num_split_seq
            # percentage of coreset to all patch features
            self.percentage_coreset = args.percentage_coreset
            # dimension to project features for sampling
            self.dim_sampling = args.dim_sampling
            # number of samples to initially randomly select coreset
            self.num_initial_coreset = args.num_initial_coreset

            # input format related
            self.shape_stretch = (args.size_crop[0], args.size_crop[1])  # (H, W)
        else:
            self.dim_coreset_feat = None
            self.num_split_seq = None
            self.percentage_coreset = None
            self.dim_sampling = None
            self.num_initial_coreset = None
            self.shape_stretch = None

        # number of nearest neighbor to get patch images
        self.k = args.k

        # consideration for the outer edge
        self.pixel_outer_decay = args.pixel_outer_decay

        # output path of trained something 
        self.path_trained = args.path_trained

    def follow(self, args_trained):
        # dimension after layer feature merging (at 2nd adaptive average pooling)
        self.dim_coreset_feat = args_trained['dim_merge_feat']
        # input format related
        self.shape_stretch = (args_trained['size_crop'][0],
                              args_trained['size_crop'][1])  # (H, W)


class ConfigDraw:
    def __init__(self, args, mode_train=True):
        # output detail or not (take a long time...)
        self.verbose = args.verbose

        # value for normalization of visualizing
        self.score_max = args.score_max

        # output filename related
        self.k = args.k

        if mode_train:
            # output filename related
            self.percentage_coreset = args.percentage_coreset

            # receptive field size
            self.size_receptive = args.size_receptive

            # aspect_ratio of output figure
            self.aspect_figure = args.size_crop[1] / args.size_crop[0]  # W / H
            self.aspect_figure = np.round(self.aspect_figure, decimals=1)

            # visualize mode
            self.mode_visualize = args.mode_visualize
            self.mode_video = False
        else:
            self.percentage_coreset = None
            self.size_receptive = None
            self.aspect_figure = None
            self.mode_visualize = 'infer'
            if args.path_video is None:
                self.mode_video = False
            else:
                self.mode_video = True
                capture = cv2.VideoCapture(args.path_video)
                self.fps_video = capture.get(cv2.CAP_PROP_FPS)
                capture.release()

        # output path of figure
        self.path_result = args.path_result

    def follow(self, args_trained):
        # output filename related
        self.percentage_coreset = args_trained['percentage_coreset']
        # receptive field size
        self.size_receptive = args_trained['size_receptive']
        # aspect_ratio of output figure
        self.aspect_figure = args_trained['size_crop'][1] / args_trained['size_crop'][0]  # W / H
        self.aspect_figure = np.round(self.aspect_figure, decimals=1)
