import numpy as np
import numba
from tqdm import tqdm
from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score


def calc_imagewise_metrics(D, type_normal='good'):
    D_list = []
    y_list = []

    pbar = tqdm(total=int(len(D.keys()) + 3),
                desc='calculate imagewise metrics')
    for type_test in D.keys():
        for i in range(len(D[type_test])):
            D_tmp = np.max(D[type_test][i])
            y_tmp = int(type_test != type_normal)

            D_list.append(D_tmp)
            y_list.append(y_tmp)
        pbar.update(1)

    D_flatten_list = np.array(D_list).reshape(-1)
    y_flatten_list = np.array(y_list).reshape(-1)
    pbar.update(1)

    fpr, tpr, _ = roc_curve(y_flatten_list, D_flatten_list)
    pbar.update(1)

    rocauc = roc_auc_score(y_flatten_list, D_flatten_list)
    pbar.update(1)
    pbar.close()

    return fpr, tpr, rocauc


def calc_pixelwise_metrics(D, y):
    D_list = []
    y_list = []

    pbar = tqdm(total=int(len(D.keys()) + 3),
                desc='calculate pixelwise metrics')
    for type_test in D.keys():
        for i in range(len(D[type_test])):
            D_tmp = D[type_test][i]
            y_tmp = y[type_test][i]

            D_list.append(D_tmp)
            y_list.append(y_tmp)
        pbar.update(1)

    D_flatten_list = np.array(D_list).reshape(-1)
    pbar.update(1)

    y_flatten_list = np.array(y_list).reshape(-1)
    pbar.update(1)

    fpr, tpr, _ = roc_curve(y_flatten_list, D_flatten_list)
    rocauc = roc_auc_score(y_flatten_list, D_flatten_list)
    pbar.update(1)
    pbar.close()

    return fpr, tpr, rocauc


# https://github.com/diditforlulz273/fastauc/blob/main/fastauc/fast_auc.py
def roc_auc_score(y_true, y_score):
    # binary clf curve
    y_true = (y_true == 1)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # roc
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0 or tps[-1] <= 0:
        return np.nan

    # auc
    direction = 1
    dx = np.diff(fps)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            return 'error'

    area = direction * np.trapz(tps, fps) / (tps[-1] * fps[-1])

    return area
