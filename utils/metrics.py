import numpy as np
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve, precision_recall_curve


def calc_imagewise_metrics(D, type_normal='good'):
    D_list = []
    y_list = []

    pbar = tqdm(total=int(len(D.keys()) + 5),
                desc='calculate imagewise metrics')
    for type_test in D.keys():
        for i in range(len(D[type_test])):
            D_tmp = np.max(D[type_test][i])
            y_tmp = int(type_test != type_normal)

            D_list.append(D_tmp)
            y_list.append(y_tmp)
        pbar.update(1)

    D_flat_list = np.array(D_list).reshape(-1)
    y_flat_list = np.array(y_list).reshape(-1)
    pbar.update(1)

    fpr, tpr, _ = roc_curve(y_flat_list, D_flat_list)
    pbar.update(1)

    rocauc = auc(fpr, tpr)
    pbar.update(1)

    pre, rec, _ = precision_recall_curve(y_flat_list, D_flat_list)
    pbar.update(1)

    # https://sinyi-chou.github.io/python-sklearn-precision-recall/
    prauc = auc(rec, pre)
    pbar.update(1)
    pbar.close()

    return fpr, tpr, rocauc, pre, rec, prauc


def calc_pixelwise_metrics(D, y):
    D_list = []
    y_list = []

    pbar = tqdm(total=int(len(D.keys()) + 6),
                desc='calculate pixelwise metrics')
    for type_test in D.keys():
        for i in range(len(D[type_test])):
            D_tmp = D[type_test][i]
            y_tmp = y[type_test][i]

            D_list.append(D_tmp)
            y_list.append(y_tmp)
        pbar.update(1)

    D_flat_list = np.array(D_list).reshape(-1)
    y_flat_list = np.array(y_list).reshape(-1)
    pbar.update(1)

    fpr, tpr, _ = roc_curve(y_flat_list, D_flat_list)
    pbar.update(1)

    rocauc = auc(fpr, tpr)
    pbar.update(1)

    pre, rec, thresh = precision_recall_curve(y_flat_list, D_flat_list)
    pbar.update(1)

    prauc = auc(rec, pre)
    pbar.update(1)

    # https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master/blob/main/main.py#L193C1-L200C1
    # get optimal threshold
    a = 2 * pre * rec
    b = pre + rec
    f1 = np.divide(a, b, out=np.zeros_like(a), where=(b != 0))
    i_opt = np.argmax(f1)
    thresh_opt = thresh[i_opt]
    pbar.update(1)
    pbar.close()

    print('pixelwise optimal threshold:%.3f (precision:%.3f, recall:%.3f)' %
          (thresh_opt, pre[i_opt], rec[i_opt]))

    return fpr, tpr, rocauc, pre, rec, prauc, thresh_opt
