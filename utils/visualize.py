import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


# https://github.com/gsurma/cnn_explainer/blob/main/utils.py
def overlay_heatmap_on_image(img, heatmap, ratio_img=0.5):
    img = img.astype(np.float32)

    heatmap = 1 - np.clip(heatmap, 0, 1)
    heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32)

    overlay = (img * ratio_img) + (heatmap * (1 - ratio_img))
    overlay = np.clip(overlay, 0, 255)
    overlay = overlay.astype(np.uint8)
    return overlay


def draw_distance_graph(type_data, cfg_draw, D, rocauc_img):
    D_list = {}
    for type_test in D.keys():
        D_list[type_test] = []
        for i in range(len(D[type_test])):
            D_tmp = np.max(D[type_test][i])
            D_list[type_test].append(D_tmp)
        D_list[type_test] = np.array(D_list[type_test])

    plt.figure(figsize=(10, 8), dpi=100, facecolor='white')

    # 'good' 1st
    N_test = 0
    type_test = 'good'
    plt.subplot(2, 1, 1)
    plt.scatter((np.arange(len(D_list[type_test])) + N_test), D_list[type_test],
                alpha=0.5, label=type_test)
    plt.subplot(2, 1, 2)
    plt.hist(D_list[type_test], alpha=0.5, label=type_test, bins=10)

    # other than 'good'
    N_test += len(D_list[type_test])
    types_test = np.array([k for k in D_list.keys() if k != 'good'])
    for type_test in types_test:
        plt.subplot(2, 1, 1)
        plt.scatter((np.arange(len(D_list[type_test])) + N_test), D_list[type_test], alpha=0.5, label=type_test)
        plt.subplot(2, 1, 2)
        plt.hist(D_list[type_test], alpha=0.5, label=type_test, bins=10)
        N_test += len(D_list[type_test])

    plt.subplot(2, 1, 1)
    plt.title('imagewise anomaly detection accuracy (ROCAUC %%) : %.3f' % rocauc_img)
    plt.grid()
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.grid()
    plt.legend()
    plt.gcf().tight_layout()
    plt.gcf().savefig(os.path.join(cfg_draw.path_result, type_data,
                                   ('pred-dist_%s_p%04d_k%02d_r%04d.png' %
                                    (type_data, (cfg_draw.percentage_coreset * 1000),
                                     cfg_draw.k, round(rocauc_img * 1000)))))
    plt.clf()
    plt.close()


def draw_heatmap(type_data, cfg_draw, D, y, D_max, imgs_test, files_test, idx_coreset, I, imgs_train, HW_map):
    for type_test in D.keys():
        for i in tqdm(range(len(D[type_test])), desc='[verbose mode] visualize localization (case:%s)' % type_test):
            file = files_test[type_test][i]
            img = imgs_test[type_test][i]
            score_map = D[type_test][i]
            idx_patch = idx_coreset[I[type_test][i][:, 0]] if idx_coreset is not None else None
            score_max = D_max
            gt = y[type_test][i] if y != {} else None

            img_patch = pickup_patch(idx_patch, imgs_train, HW_map, cfg_draw.size_receptive_field) \
                if imgs_train is not None \
                else None

            fig_width = 10 * max(1, cfg_draw.aspect_figure)
            fig_height = height = 18 if cfg_draw.mode_visualize == 'eval' else 10
            plt.figure(figsize=(fig_width, fig_height), dpi=100, facecolor='white')
            plt.rcParams['font.size'] = 10

            if cfg_draw.mode_visualize == 'eval':
                plt.subplot2grid((7, 3), (0, 0), rowspan=1, colspan=1)
                plt.imshow(img)
                plt.title('%s : %s' % (file.split('/')[-2], file.split('/')[-1]))

                plt.subplot2grid((7, 3), (0, 1), rowspan=1, colspan=1)
                plt.imshow(gt)

                plt.subplot2grid((7, 3), (0, 2), rowspan=1, colspan=1)
                plt.imshow(score_map)
                plt.colorbar()
                plt.title('max score : %.2f' % score_max)

                plt.subplot2grid((42, 2), (7, 0), rowspan=10, colspan=1)
                plt.imshow(overlay_heatmap_on_image(img, (score_map / score_max)))

                plt.subplot2grid((42, 2), (7, 1), rowspan=10, colspan=1)
                plt.imshow((img.astype(np.float32) * (score_map / score_max)[..., None]).astype(np.uint8))

                plt.subplot2grid((21, 1), (10, 0), rowspan=11, colspan=1)
                plt.imshow(img_patch, interpolation='none')
                plt.title('patch images created with top1-NN')
            elif cfg_draw.mode_visualize == 'infer':
                plt.subplot2grid((10, 2), (0, 0), rowspan=4, colspan=1)
                plt.imshow(img)
                plt.title('%s : %s' % (file.split('/')[-2], file.split('/')[-1]))

                plt.subplot2grid((10, 2), (0, 1), rowspan=4, colspan=1)
                plt.imshow(score_map)
                plt.colorbar()
                plt.title('max score : %.2f' % score_max)

                plt.subplot2grid((10, 2), (5, 0), rowspan=4, colspan=1)
                plt.imshow(overlay_heatmap_on_image(img, (score_map / score_max)))

                plt.subplot2grid((10, 2), (5, 1), rowspan=4, colspan=1)
                plt.imshow((img.astype(np.float32) * (score_map / score_max)[..., None]).astype(np.uint8))

            score_tmp = np.max(score_map) / score_max * 100
            filename_out = os.path.join(cfg_draw.path_result, type_data,
                                        ('localization_%s_%s_%s_p%04d_k%02d_s%03d.png' %
                                         (type_data, type_test, 
                                          os.path.basename(file).split('.')[0],
                                          (cfg_draw.percentage_coreset * 1000),
                                          cfg_draw.k, round(score_tmp))))

            plt.gcf().savefig(filename_out)
            plt.clf()
            plt.close()


def pickup_patch(idx_patch, imgs, HW_map, size_receptive_field):
    # get input shape
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

    # build blank image for output
    img_patch = np.zeros([(HW_map[0] * size_receptive_field),
                          (HW_map[1] * size_receptive_field), 3], dtype=np.uint8)
    i_y = 0
    i_x = 0

    # loop of patch feature index
    for i_patch in idx_patch:
        i_img = i_patch // (HW_map[0] * HW_map[1])
        i_HW = i_patch % (HW_map[0] * HW_map[1])
        i_H = i_HW // HW_map[1]
        i_W = i_HW % HW_map[1]

        img = imgs[i_img]
        y = y_pitch[i_H]
        x = x_pitch[i_W]
        img_piece = img[(y - h_half):(y + h_half + 1),
                        (x - w_half):(x + w_half + 1)]

        y = i_y * size_receptive_field
        x = i_x * size_receptive_field
        img_patch[y:(y + size_receptive_field),
                  x:(x + size_receptive_field)] = img_piece
        i_x += 1
        if (i_x >= HW_map[1]):
            i_x = 0
            i_y += 1

    return img_patch


def draw_roc_curve(cfg_draw, fpr_img, tpr_img, rocauc_img, rocauc_img_mean,
                             fpr_pix, tpr_pix, rocauc_pix, rocauc_pix_mean):
    plt.figure(figsize=(12, 6), dpi=100, facecolor='white')
    for type_data in fpr_img.keys():
        plt.subplot(1, 2, 1)
        plt.plot(fpr_img[type_data], tpr_img[type_data],
                 label='%s ROCAUC: %.3f' % (type_data, rocauc_img[type_data]))
        plt.subplot(1, 2, 2)
        plt.plot(fpr_pix[type_data], tpr_pix[type_data],
                 label='%s ROCAUC: %.3f' % (type_data, rocauc_pix[type_data]))

    plt.subplot(1, 2, 1)
    plt.title('imagewise anomaly detection accuracy (ROCAUC %%) : mean %.3f' % rocauc_img_mean)
    plt.grid()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('pixelwise anomaly detection accuracy (ROCAUC %%) : mean %.3f' % rocauc_pix_mean)
    plt.grid()
    plt.legend()
    plt.gcf().tight_layout()
    plt.gcf().savefig(os.path.join(cfg_draw.path_result,
                                   ('roc-curve_p%04d_k%02d_rim%04d_rpm%04d.png' %
                                    ((cfg_draw.percentage_coreset * 1000), cfg_draw.k,
                                     round(rocauc_img_mean * 1000),
                                     round(rocauc_pix_mean * 1000)))))
    plt.clf()
    plt.close()
