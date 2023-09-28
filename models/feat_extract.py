import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision
from torchinfo import summary


class FeatExtract:
    def __init__(self, cfg_feat):
        self.device = cfg_feat.device
        self.batch_size = cfg_feat.batch_size
        self.shape_input = cfg_feat.SHAPE_INPUT
        self.layer_map = cfg_feat.layer_map
        self.size_patch = cfg_feat.size_patch
        self.dim_each_feat = cfg_feat.dim_each_feat
        self.dim_merge_feat = cfg_feat.dim_merge_feat
        self.MEAN = cfg_feat.MEAN
        self.STD = cfg_feat.STD

        self.padding = int((self.size_patch - 1) / 2)
        self.stride = 1  # fixed temporarily...

        code = 'self.backbone = %s(weights=%s)' % (cfg_feat.backbone, cfg_feat.weight)
        exec(code)
        self.backbone.eval()
        self.backbone.to(self.device)
        summary(self.backbone, input_size=(1, 3, *self.shape_input))

        self.feat = []
        for layer_map in self.layer_map:
            code = 'self.backbone.%s.register_forward_hook(self.hook)' % layer_map
            exec(code)

        # dummy forward
        x = torch.zeros(1, 3, self.shape_input[0], self.shape_input[1])  # RGB
        x = x.to(self.device)
        self.feat = []
        with torch.no_grad():
            _ = self.backbone(x)

        # https://github.com/amazon-science/patchcore-inspection/blob/main/src/patchcore/patchcore.py#L295
        self.unfolder = torch.nn.Unfold(kernel_size=self.size_patch, stride=self.stride,
                                        padding=self.padding, dilation=1)
        self.patch_shapes = []
        for i in range(len(self.feat)):
            number_of_total_patches = []
            for s in self.feat[i].shape[-2:]:
                n_patches = (
                    s + 2 * self.padding - 1 * (self.size_patch - 1) - 1
                ) / self.stride + 1
                number_of_total_patches.append(int(n_patches))
            self.patch_shapes.append(number_of_total_patches)

    def hook(self, module, input, output):
        self.feat.append(output.detach().cpu())

    def HW_map(self):
        return self.patch_shapes[0]

    def normalize(self, input):
        x = torch.from_numpy(input.astype(np.float32))
        x = x.to(self.device)
        x = x / 255
        x = x - self.MEAN
        x = x / self.STD
        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        return x

    # return : is_train=True->torch.Tensor, is_train=False->dict
    def extract(self, imgs, is_train=True):
        if is_train:
            # feature extract for train and aggregate split-image for explain
            x_batch = []
            self.feat = []
            for i, img in tqdm(enumerate(imgs),
                               desc='extract feature for train (case:good)'):
                x = self.normalize(img)
                x_batch.append(x)

                if ((len(x_batch) == self.batch_size) | (i == (len(imgs) - 1))):
                    with torch.no_grad():
                        _ = self.backbone(torch.vstack(x_batch))
                    x_batch = []

            feat = []
            num_layer = len(self.layer_map)
            for i_layer_map in range(num_layer):
                feat.append(torch.vstack(self.feat[i_layer_map::num_layer]))
            feat = self.patchfy(feat)
        else:
            # feature extract for test
            feat = {}
            for type_test in imgs.keys():
                x_batch = []
                self.feat = []
                for i, img in tqdm(enumerate(imgs[type_test]),
                                   desc='extract feature for test (case:%s)' % type_test):
                    x = self.normalize(img)
                    x_batch.append(x)

                    if ((len(x_batch) == self.batch_size) | (i == (len(imgs[type_test]) - 1))):
                        with torch.no_grad():
                            _ = self.backbone(torch.vstack(x_batch))
                        x_batch = []

                feat[type_test] = []
                num_layer = len(self.layer_map)
                for i_layer_map in range(num_layer):
                    feat[type_test].append(torch.vstack(self.feat[i_layer_map::num_layer]))
                feat[type_test] = self.patchfy(feat[type_test])
        return feat

    def patchfy(self, feat, batch_size_patchfy=2000):
        pbar = tqdm(total=((len(feat) * 3) + 1), desc='patchfy feature')

        with torch.no_grad():
            # unfold
            for i in range(len(feat)):
                _feat = feat[i]
                BC_before_unfold = _feat.shape[:2]
                # (B, C, H, W) -> (B, CPHPW, HW)
                # print('_feat.shape[:4] =', _feat.shape[:4])
                # __feat = torch.zeros([*_feat.shape[:4], self.size_patch, self.size_patch])
                # for i_batch in range(0, len(_feat), batch_size_patchfy):
                #     print('i_batch =', i_batch)
                #     feat_tmp = _feat[i_batch:(i_batch + batch_size_patchfy)]
                #     feat_tmp = self.unfolder(feat_tmp)
                #     __feat[i_batch:(i_batch + batch_size_patchfy)] = feat_tmp.cpu()
                # _feat = __feat
                _feat = self.unfolder(_feat)
                # (B, CPHPW, HW) -> (B, C, PH, PW, HW)
                _feat = _feat.reshape(*BC_before_unfold,
                                      self.size_patch, self.size_patch, -1)
                # (B, C, PH, PW, HW) -> (B, HW, C, PW, HW)
                _feat = _feat.permute(0, 4, 1, 2, 3)
                feat[i] = _feat
                pbar.update(1)

            # expand small feat to fit large features
            for i in range(1, len(feat)):
                _feat = feat[i]
                patch_dims = self.patch_shapes[i]
                # (B, HW, C, PW, HW) -> (B, H, W, C, PH, PW)
                _feat = _feat.reshape(_feat.shape[0], patch_dims[0],
                                      patch_dims[1], *_feat.shape[2:])
                # (B, H, W, C, PH, PW) -> (B, C, PH, PW, H, W)
                _feat = _feat.permute(0, -3, -2, -1, 1, 2)
                perm_base_shape = _feat.shape
                # (B, C, PH, PW, H, W) -> (BCPHPW, H, W)
                _feat = _feat.reshape(-1, *_feat.shape[-2:])
                # (BCPHPW, H, W) -> (BCPHPW, H_max, W_max)
                __feat = torch.zeros([len(_feat), self.HW_map()[0], self.HW_map()[1]])
                for i_batch in range(0, len(_feat), batch_size_patchfy):
                    feat_tmp = _feat[i_batch:(i_batch + batch_size_patchfy)]
                    feat_tmp = feat_tmp.unsqueeze(1).to(self.device)
                    feat_tmp = F.interpolate(feat_tmp,
                                             size=(self.HW_map()[0], self.HW_map()[1]),
                                             mode="bilinear", align_corners=False)
                    __feat[i_batch:(i_batch + batch_size_patchfy)] = feat_tmp.squeeze(1).cpu()
                _feat = __feat
                # _feat = F.interpolate(_feat.unsqueeze(1),
                #                       size=(self.HW_map()[0], self.HW_map()[1]),
                #                       mode="bilinear", align_corners=False)
                # _feat = _feat.squeeze(1)
                # for i_batch in range(0, len(_feat), 10000):
                #     print(torch.sum(torch.abs(_feat[i_batch:(i_batch + 10000)] - __feat[i_batch:(i_batch + 10000)])))
                # (BCPHPW, H_max, W_max) -> (B, C, PH, PW, H_max, W_max)
                _feat = _feat.reshape(*perm_base_shape[:-2],
                                      self.HW_map()[0], self.HW_map()[1])
                # (B, C, PH, PW, H_max, W_max) -> (B, H_max, W_max, C, PH, PW)
                _feat = _feat.permute(0, -2, -1, 1, 2, 3)
                # (B, H_max, W_max, C, PH, PW) -> (B, H_maxW_max, C, PH, PW)
                _feat = _feat.reshape(len(_feat), -1, *_feat.shape[-3:])
                feat[i] = _feat
                pbar.update(1)

            # aggregate feature vectors
            # (B, H, W, C, PH, PW) -> (BHW, C, PH, PW)
            feat = [x.reshape(-1, *x.shape[-3:]) for x in feat]
            pbar.update(1)

            # adaptive average pooling for each feature vector
            for i in range(len(feat)):
                _feat = feat[i]
                # (BHW, C, PH, PW) -> (BHW, 1, CPHPW)
                _feat = _feat.reshape(len(_feat), 1, -1)
                # (BHW, 1, CPHPW) -> (BHW, D_e)
                _feat = F.adaptive_avg_pool1d(_feat,
                                              self.dim_each_feat).squeeze(1)
                feat[i] = _feat
                pbar.update(1)

            # concat the two feature vectors and adaptive average pooling
            # (BHW, D_e) -> (BHW, D_e*2)
            feat = torch.stack(feat, dim=1)
            """Returns reshaped and average pooled feat."""
            # batchsize x number_of_layers x input_dim -> batchsize x target_dim
            # (BHW, D_e*2) -> (BHW, D_m)
            feat = feat.reshape(len(feat), 1, -1)
            feat = F.adaptive_avg_pool1d(feat, self.dim_merge_feat)
            feat = feat.reshape(len(feat), -1)
            pbar.update(1)

        pbar.close()
        return feat
