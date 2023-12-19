import os

import numpy as np
import cv2
from tqdm import tqdm
import torch
import faiss
from scipy.ndimage import gaussian_filter


class PatchCore:
    def __init__(self, cfg_patchcore, HW_map):
        self.device = cfg_patchcore.device
        self.k = cfg_patchcore.k
        self.dim_coreset_feat = cfg_patchcore.dim_coreset_feat
        self.num_split_seq = cfg_patchcore.num_split_seq
        self.percentage_coreset = cfg_patchcore.percentage_coreset
        self.dim_sampling = cfg_patchcore.dim_sampling
        self.num_initial_coreset = cfg_patchcore.num_initial_coreset
        self.seed = cfg_patchcore.seed
        self.shape_stretch = cfg_patchcore.shape_stretch
        self.pixel_outer_decay = cfg_patchcore.pixel_outer_decay
        self.HW_map = HW_map

        # prep knn index
        if self.device.type != 'cuda':
            self.index_feat = faiss.IndexFlatL2(self.dim_coreset_feat)
        else:
            self.index_feat = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                                   self.dim_coreset_feat,
                                                   faiss.GpuIndexFlatConfig())

        # prep mapper
        self.mapper = torch.nn.Linear(self.dim_coreset_feat, self.dim_sampling,
                                      bias=False).to(self.device)

        self.faiss_save_dir = cfg_patchcore.faiss_save_dir
        os.makedirs(self.faiss_save_dir, exist_ok=True)

        self.coreset_patch_save_dir = cfg_patchcore.coreset_patch_save_dir
        os.makedirs(self.coreset_patch_save_dir, exist_ok=True)

    def compute_greedy_coreset_idx(self, feat):
        feat = feat.to(self.device)
        with torch.no_grad():
            feat_proj = self.mapper(feat)

        _num_initial_coreset = np.clip(self.num_initial_coreset,
                                       None, len(feat_proj))
        np.random.seed(self.seed)
        start_points = np.random.choice(len(feat_proj), _num_initial_coreset,
                                        replace=False).tolist()

        # computes batchwise Euclidean distances using PyTorch
        mat_A = feat_proj
        mat_B = feat_proj[start_points]
        A_x_A = mat_A.unsqueeze(1).bmm(mat_A.unsqueeze(2)).reshape(-1, 1)
        B_x_B = mat_B.unsqueeze(1).bmm(mat_B.unsqueeze(2)).reshape(1, -1)
        A_x_B = mat_A.mm(mat_B.T)
        # not need sqrt
        mat_dist = (-2 * A_x_B + A_x_A + B_x_B).clamp(0, None)

        dist_coreset_anchor = torch.mean(mat_dist, axis=-1, keepdims=True)

        idx_coreset = []
        num_coreset_samples = int(len(feat_proj) * self.percentage_coreset)

        with torch.no_grad():
            for _ in tqdm(range(num_coreset_samples), desc="sampling"):
                idx_select = torch.argmax(dist_coreset_anchor).item()
                idx_coreset.append(idx_select)

                mat_A = feat_proj
                mat_B = feat_proj[[idx_select]]
                # computes batchwise Euclidean distances using PyTorch
                A_x_A = mat_A.unsqueeze(1).bmm(mat_A.unsqueeze(2)).reshape(-1, 1)
                B_x_B = mat_B.unsqueeze(1).bmm(mat_B.unsqueeze(2)).reshape(1, -1)
                A_x_B = mat_A.mm(mat_B.T)
                # not need sqrt
                mat_select_dist = (-2 * A_x_B + A_x_A + B_x_B).clamp(0, None)

                dist_coreset_anchor = torch.cat([dist_coreset_anchor,
                                                 mat_select_dist], dim=-1)
                dist_coreset_anchor = torch.min(dist_coreset_anchor,
                                                dim=1).values.reshape(-1, 1)

        idx_coreset = np.array(idx_coreset)
        return idx_coreset

    def reset_neighbor(self):
        self.index_feat.reset()

    def add_neighbor(self, feat_train):
        self.index_feat.add(feat_train.numpy())

    def save_neighbor(self, type_data):
        faiss_save_path = os.path.join(self.faiss_save_dir, f'{type_data}.idx')
        cpu_index = faiss.index_gpu_to_cpu(self.index_feat)
        faiss.write_index(cpu_index, faiss_save_path)

    def load_neighbor(self, type_data):
        faiss_idx_path = os.path.join(self.faiss_save_dir, f'{type_data}.idx')
        self.index_feat = faiss.read_index(faiss_idx_path)

    def load_neighbor_from_file(self, file_path):
        self.index_feat = faiss.read_index(file_path)

    def localization(self, feat_test, show_progress=True):
        D = {}
        D_max = -9999
        I = {}
        # loop for test cases
        for type_test in feat_test.keys():
            D[type_test] = []
            I[type_test] = []

            # loop for test data
            _feat_test = feat_test[type_test]
            _feat_test = _feat_test.reshape(-1, (self.HW_map[0] * self.HW_map[1]), self.dim_coreset_feat)
            _feat_test = _feat_test.numpy()
            num_data = len(_feat_test)
            for i in tqdm(range(num_data), desc='localization (case:%s)' % type_test, disable=not show_progress):
                # measure distance pixelwise
                score_map, I_tmp = self.measure_dist_pixelwise(feat_test=_feat_test[i])
                # adjust score of outer-pixel (provisional heuristic algorithm)
                if self.pixel_outer_decay > 0:
                    score_map[:self.pixel_outer_decay, :] *= 0.6
                    score_map[-self.pixel_outer_decay:, :] *= 0.6
                    score_map[:, :self.pixel_outer_decay] *= 0.6
                    score_map[:, -self.pixel_outer_decay:] *= 0.6
                # stock score map
                D[type_test].append(score_map)
                D_max = max(D_max, np.max(score_map))
                I[type_test].append(I_tmp)

            # cast list to numpy array
            D[type_test] = np.array(D[type_test])
            I[type_test] = np.array(I[type_test])

        return D, D_max, I

    def measure_dist_pixelwise(self, feat_test):
        # k nearest neighbor
        D, I = self.index_feat.search(feat_test, self.k)
        D = np.mean(D, axis=-1)

        # transform to scoremap
        score_map = D.reshape(*self.HW_map)
        score_map = cv2.resize(score_map, (self.shape_stretch[1], self.shape_stretch[0]))

        # apply gaussian smoothing on the score map
        score_map_smooth = gaussian_filter(score_map, sigma=4)

        return score_map_smooth, I

    def pickup_patch(self, idx_patch, imgs, HW_map, size_receptive_field):
        h = imgs.shape[-3]
        w = imgs.shape[-2]

        # calculate half size for split
        h_half = int((size_receptive_field - 1) / 2)
        w_half = int((size_receptive_field - 1) / 2)

        # calculate center-coordinates of split-image
        y_pitch = np.arange(0, (h - 1 + 1e-10), ((h - 1) / (HW_map[0] - 1)))
        y_pitch = np.round(y_pitch).astype(np.int16)
        y_pitch = y_pitch + h_half
        x_pitch = np.arange(0, (w - 1 + 1e-10), ((w - 1) / (HW_map[1] - 1)))
        x_pitch = np.round(x_pitch).astype(np.int16)
        x_pitch = x_pitch + w_half
        # padding to normal images
        imgs = np.pad(imgs, ((0, 0), (h_half, h_half), (w_half, w_half), (0, 0)))

        img_piece_list = []
        for i_patch in idx_patch:
            i_img = i_patch // (HW_map[0] * HW_map[1])
            i_HW = i_patch % (HW_map[0] * HW_map[1])
            i_H = i_HW // HW_map[1]
            i_W = i_HW % HW_map[1]

            img = imgs[i_img]
            y = y_pitch[i_H]
            x = x_pitch[i_W]
            img_piece = img[(y - h_half):(y + h_half + 1), (x - w_half):(x + w_half + 1)]
            img_piece_list.append(img_piece)

        img_piece_array = np.stack(img_piece_list)

        return img_piece_array

    def save_coreset_patch(self, idx_coreset, type_data, image_train, HW_map, cfg_draw):
        img_patch = self.pickup_patch(idx_coreset, image_train, HW_map, cfg_draw.size_receptive_field)
        coreset_img_file_path = os.path.join(self.coreset_patch_save_dir, f'{type_data}_coreset_patch_img.npy')
        np.save(coreset_img_file_path, img_patch)

        coreset_idx_file_path = os.path.join(self.coreset_patch_save_dir, f'{type_data}_coreset_patch_idx.npy')
        np.save(coreset_idx_file_path, idx_coreset)

    def load_coreset_patch(self, type_data):
        coreset_img_file_path = os.path.join(self.coreset_patch_save_dir, f'{type_data}_coreset_patch_img.npy')
        coreset_patch_img = np.load(coreset_img_file_path)

        coreset_idx_file_path = os.path.join(self.coreset_patch_save_dir, f'{type_data}_coreset_patch_idx.npy')
        coreset_patch_idx = np.load(coreset_idx_file_path)

        return coreset_patch_idx, coreset_patch_img

    def load_coreset_patch_from_file(self, file_path):
        coreset_patch_img = np.load(file_path)
        return coreset_patch_img

