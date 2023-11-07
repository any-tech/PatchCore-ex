import os
import numpy as np
import torch

# https://pytorch.org/vision/main/models/generated/torchvision.models.wide_resnet50_2.html
MEAN = torch.FloatTensor([[[0.485, 0.456, 0.406]]])
STD = torch.FloatTensor([[[0.229, 0.224, 0.225]]])


class ConfigData:
    @classmethod
    def __init__(cls, args):
        # file reading related
        cls.path_parent = args.path_parent
        assert os.path.exists(cls.path_parent)
        cls.num_cpu_max = args.num_cpu_max
        cls.shuffle = (args.num_split_seq > 1)  # for k-center-greedy split
        cls.seed = args.seed

        # input format related
        cls.SHAPE_MIDDLE = (args.size_resize[0], args.size_resize[1])  # (H, W)
        cls.SHAPE_INPUT = (args.size_crop[0], args.size_crop[1])  # (H, W)
        cls.pixel_cut = (int((cls.SHAPE_MIDDLE[0] - cls.SHAPE_INPUT[0]) / 2),
                         int((cls.SHAPE_MIDDLE[1] - cls.SHAPE_INPUT[1]) / 2))  # (H, W)

        # augmantation related
        cls.flip_horz = args.flip_horz
        cls.flip_vert = args.flip_vert

        # collect types of data
        if args.types_data is None:
            types_data = [d for d in os.listdir(args.path_parent)
                          if os.path.isdir(os.path.join(args.path_parent, d))]
            cls.types_data = np.sort(np.array(types_data))
        else:
            for type_data in args.types_data:
                assert os.path.exists(os.path.join(args.path_parent, type_data))
            cls.types_data = np.sort(np.array(args.types_data))


class ConfigFeat:
    def __init__(self, args):
        # adjsut to environment
        if args.cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

        # batch-size for feature extraction by ImageNet model
        self.batch_size = args.batch_size

        # input format related
        self.SHAPE_INPUT = (args.size_crop[0], args.size_crop[1])  # (H, W)

        # base network
        self.backbone = args.backbone
        self.weight = args.weight

        # layer specification
        self.layer_map = args.layer_map

        # patch pixel of feature map for increasing receptive field size
        self.size_patch = args.size_patch

        # dimension of each layer feature (at 1st adaptive average pooling)
        self.dim_each_feat = args.dim_each_feat
        # dimension after layer feature merging (at 2nd adaptive average pooling)
        self.dim_merge_feat = args.dim_merge_feat

        # adjust to the network's learning policy and the data conditions
        self.MEAN = MEAN.to(self.device)
        self.STD = STD.to(self.device)

        self.merge_dst_layer = args.merge_dst_layer


class ConfigPatchCore:
    def __init__(self, args):
        # adjsut to environment
        if args.cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

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
        # random-seed for k-center-greedy
        self.seed = args.seed

        # number of nearest neighbor to get patch images
        self.k = args.k

        # input format related
        self.shape_stretch = (args.size_crop[0], args.size_crop[1])  # (H, W)

        # consideration for the outer edge
        self.pixel_outer_decay = args.pixel_outer_decay

        self.faiss_save_dir = args.faiss_save_dir


class ConfigDraw:
    def __init__(self, args):
        # output detail or not (take a long time...)
        self.verbose = args.verbose

        # output filename related
        self.k = args.k
        self.percentage_coreset = args.percentage_coreset

        # receptive field size
        self.size_receptive_field = args.size_receptive_field

        # aspect_ratio of output figure
        self.aspect_figure = args.size_crop[1] / args.size_crop[0]  # W / H
        self.aspect_figure = np.round(self.aspect_figure, decimals=1)

        # visualize mode
        self.mode_visualize = args.mode_visualize

        # output path of figure
        self.path_result = args.path_result
