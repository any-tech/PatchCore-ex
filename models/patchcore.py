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
        self.index_feat = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                               self.dim_coreset_feat,
                                               faiss.GpuIndexFlatConfig())

        # prep mapper
        self.mapper = torch.nn.Linear(self.dim_coreset_feat, self.dim_sampling,
                                      bias=False).to(self.device)

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

    def localization(self, feat_test):
        D = {}
        D_max = -9999
        I = {}
        # loop for test cases
        for type_test in feat_test.keys():
            D[type_test] = []
            I[type_test] = []

            # loop for test data
            _feat_test = feat_test[type_test]
            _feat_test = _feat_test.reshape(-1, (self.HW_map[0] * self.HW_map[1]),
                                            self.dim_coreset_feat)
            _feat_test = _feat_test.numpy()
            num_data = len(_feat_test)
            for i in tqdm(range(num_data),
                          desc='localization (case:%s)' % type_test):
                # measure distance pixelwise
                score_map, I_tmp = self.measure_dist_pixelwise(feat_test=_feat_test[i])
                # adjust score of outer-pixel (provisional heuristic algorithm)
                if (self.pixel_outer_decay > 0):
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
        score_map = cv2.resize(score_map, (self.shape_stretch[1],
                                           self.shape_stretch[0]))

        # apply gaussian smoothing on the score map
        score_map_smooth = gaussian_filter(score_map, sigma=4)

        return score_map_smooth, I
