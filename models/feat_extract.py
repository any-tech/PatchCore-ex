import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision
from torchinfo import summary


def print_weight_device(backbone):
    for name, module in backbone.named_modules():
        if hasattr(module, 'weight') and getattr(module, 'weight') is not None:
            weight_device = module.weight.device
            weight_shape = module.weight.shape
            print(f"{name}.weight - Device: {weight_device}, Shape: {weight_shape}")

        if hasattr(module, 'bias') and getattr(module, 'bias') is not None:
            bias_device = module.bias.device
            print(f"{name}.bias - Device: {bias_device}")


class FeatExtract:
    def __init__(self, cfg_feat):
        self.device = cfg_feat.device
        self.batch_size = cfg_feat.batch_size
        self.shape_input = cfg_feat.SHAPE_INPUT
        self.layer_map = cfg_feat.layer_map
        self.layer_weights = cfg_feat.layer_weights
        self.size_patch = cfg_feat.size_patch
        self.dim_each_feat = cfg_feat.dim_each_feat
        self.dim_merge_feat = cfg_feat.dim_merge_feat
        self.MEAN = cfg_feat.MEAN
        self.STD = cfg_feat.STD
        self.merge_dst_index = self.layer_map.index(cfg_feat.merge_dst_layer)

        self.padding = int((self.size_patch - 1) / 2)
        self.stride = 1  # fixed temporarily...

        code = 'self.backbone = %s(weights=%s)' % (cfg_feat.backbone, cfg_feat.weight)
        exec(code)
        summary(self.backbone, input_size=(1, 3, *self.shape_input))
        self.backbone.eval()

        # Executing summary will force the backbone device to be changed to cuda, so do to(device) after summary
        self.backbone.to(self.device)

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
        return self.patch_shapes[self.merge_dst_index]

    def normalize(self, input):
        x = torch.from_numpy(input.astype(np.float32))
        x = x.to(self.device)
        x = x / 255
        x = x - self.MEAN
        x = x / self.STD
        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        return x

    # return : is_train=True->torch.Tensor, is_train=False->dict
    def extract(self, imgs, case='train (case:good)', batch_size_patchfy=50, show_progress=True):
        # feature extract for train and aggregate split-image for explain
        x_batch = []
        self.feat = []
        for i_img in tqdm(range(len(imgs)), desc='extract feature for %s' % case, disable=not show_progress):
            img = imgs[i_img]
            x = self.normalize(img)
            x_batch.append(x)

            if (len(x_batch) == self.batch_size) | (i_img == (len(imgs) - 1)):
                with torch.no_grad():
                    _ = self.backbone(torch.vstack(x_batch))
                x_batch = []

        # adjust
        feat = []
        num_layer = len(self.layer_map)
        for i_layer_map in range(num_layer):
            feat.append(torch.vstack(self.feat[i_layer_map::num_layer]))

        # patchfy (consider out of memory)
        num_patch_per_image = self.HW_map()[0] * self.HW_map()[1]
        num_patch = len(imgs) * num_patch_per_image
        feat_patchfy = torch.zeros(num_patch, self.dim_merge_feat)

        num_patchfy_process = (len(feat) * 3) + 1
        num_iter = np.ceil(len(imgs) / batch_size_patchfy)
        pbar = tqdm(total=int(num_patchfy_process * num_iter), desc='patchfy feature', disable=not show_progress)
        for i_batch in range(0, len(imgs), batch_size_patchfy):
            feat_tmp = []
            for feat_layer in feat:
                feat_tmp.append(feat_layer[i_batch:(i_batch + batch_size_patchfy)])
            i_from = i_batch * num_patch_per_image
            i_to = (i_batch + batch_size_patchfy) * num_patch_per_image
            feat_patchfy[i_from:i_to] = self.patchfy(feat_tmp, pbar)
        pbar.close()

        return feat_patchfy

    def patchfy(self, feat, pbar, batch_size_interp=2000):

        with torch.no_grad():
            # unfold
            for i in range(len(feat)):
                _feat = feat[i]
                BC_before_unfold = _feat.shape[:2]
                # (B, C, H, W) -> (B, CPHPW, HW)
                _feat = self.unfolder(_feat)
                # (B, CPHPW, HW) -> (B, C, PH, PW, HW)
                _feat = _feat.reshape(*BC_before_unfold,
                                      self.size_patch, self.size_patch, -1)
                # (B, C, PH, PW, HW) -> (B, HW, C, PW, HW)
                _feat = _feat.permute(0, 4, 1, 2, 3)
                feat[i] = _feat
                pbar.update(1)

            # expand small feat to fit large features
            for i in range(0, len(feat)):
                if i == self.merge_dst_index:
                    continue

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
                feat_dst = torch.zeros([len(_feat), self.HW_map()[0], self.HW_map()[1]])
                for i_batch in range(0, len(_feat), batch_size_interp):
                    feat_tmp = _feat[i_batch:(i_batch + batch_size_interp)]
                    feat_tmp = feat_tmp.unsqueeze(1).to(self.device)
                    feat_tmp = F.interpolate(feat_tmp,
                                             size=(self.HW_map()[0], self.HW_map()[1]),
                                             mode="bilinear", align_corners=False)
                    feat_dst[i_batch:(i_batch + batch_size_interp)] = feat_tmp.squeeze(1).cpu()
                _feat = feat_dst
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
                _feat = feat[i] * self.layer_weights[i]

                # (BHW, C, PH, PW) -> (BHW, 1, CPHPW)
                _feat = _feat.reshape(len(_feat), 1, -1)

                # (BHW, 1, CPHPW) -> (BHW, D_e)
                _feat = F.adaptive_avg_pool1d(_feat, self.dim_each_feat).squeeze(1)
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

        return feat
