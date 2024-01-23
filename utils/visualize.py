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
        for i_D in range(len(D[type_test])):
            D_tmp = np.max(D[type_test][i_D])
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
    plt.title('imagewise ROCAUC %% : %.3f' % rocauc_img)
    plt.grid()
    plt.legend(loc='upper left')
    plt.subplot(2, 1, 2)
    plt.grid()
    plt.legend(loc='upper right')
    plt.gcf().tight_layout()
    plt.gcf().savefig(('%s/%s/pred-dist_%s_p%04d_k%02d_rocaucimg%04d.png' %
                       (cfg_draw.path_result, type_data,
                        type_data, (cfg_draw.percentage_coreset * 1000),
                        cfg_draw.k, round(rocauc_img * 1000))))
    plt.clf()
    plt.close()


def draw_heatmap(type_data, cfg_draw, D, y, D_max, I, imgs_test, files_test,
                 imgs_coreset, HW_map):

    if cfg_draw.mode_visualize == 'eval':
        fig_width = 10 * max(1, cfg_draw.aspect_figure)
        fig_height = 18
        pixel_cut=[160, 140, 60, 60]  # [top, bottom, left right]
    else:
        fig_width = 20 * max(1, cfg_draw.aspect_figure)
        fig_height = 16
        pixel_cut=[140, 120, 180, 180]  # [top, bottom, left right]
    dpi = 100

    for type_test in D.keys():

        if cfg_draw.mode_video:
            filename_out = ('%s/%s/localization_%s_%s_p%04d_k%02d.mp4' %
                            (cfg_draw.path_result, type_data,
                             type_data, files_test[type_test][0].split('.')[0],
                             (cfg_draw.percentage_coreset * 1000), cfg_draw.k))
            # build writer
            codecs = 'mp4v'
            fourcc = cv2.VideoWriter_fourcc(*codecs)
            width = (fig_width * dpi) - pixel_cut[2] - pixel_cut[3]
            height = (fig_height * dpi) - pixel_cut[0] - pixel_cut[1]
            writer = cv2.VideoWriter(filename_out, fourcc, cfg_draw.fps_video,
                                     (width, height), True)

        desc = '[verbose mode] visualize localization (case:%s)' % type_test
        for i_D in tqdm(range(len(D[type_test])), desc=desc):
            file = files_test[type_test][i_D]
            img = imgs_test[type_test][i_D]
            score_map = D[type_test][i_D]
            if cfg_draw.score_max is None:
                score_max = D_max
            else:
                score_max = cfg_draw.score_max
            if y is not None:
                gt = y[type_test][i_D]

            I_tmp = I[type_test][i_D, :, 0]
            img_patch = assemble_patch(I_tmp, imgs_coreset, HW_map)

            plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='white')
            plt.rcParams['font.size'] = 10

            score_map_reg = score_map / score_max

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
                plt.imshow(overlay_heatmap_on_image(img, score_map_reg))

                plt.subplot2grid((42, 2), (7, 1), rowspan=10, colspan=1)
                plt.imshow((img.astype(np.float32) * score_map_reg[..., None]).astype(np.uint8))

                plt.subplot2grid((21, 1), (10, 0), rowspan=11, colspan=1)
                plt.imshow(img_patch, interpolation='none')
                plt.title('patch images created with top1-NN')

            elif cfg_draw.mode_visualize == 'infer':
                plt.subplot(2, 2, 1)
                plt.imshow(img)
                plt.title('%s : %s' % (file.split('/')[-2], file.split('/')[-1]))

                plt.subplot(2, 2, 2)
                plt.imshow(score_map)
                plt.colorbar()
                plt.title('max score : %.2f' % score_max)

                plt.subplot(2, 2, 3)
                plt.imshow(overlay_heatmap_on_image(img, score_map_reg))

                plt.subplot(2, 2, 4)
                plt.imshow(img_patch, interpolation='none')
                plt.title('patch images created with top1-NN')

            score_tmp = np.max(score_map) / score_max * 100
            plt.gcf().canvas.draw()
            img_figure = np.fromstring(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
            img_figure = img_figure.reshape(fig_height * dpi, -1, 3)
            img_figure = img_figure[pixel_cut[0]:(img_figure.shape[0] - pixel_cut[1]),
                                    pixel_cut[2]:(img_figure.shape[1] - pixel_cut[3])]

            if not cfg_draw.mode_video:
                filename_out = ('%s/%s/localization_%s_%s_%s_p%04d_k%02d_s%03d.png' %
                                (cfg_draw.path_result, type_data,
                                 type_data, type_test, os.path.basename(file).split('.')[0],
                                 (cfg_draw.percentage_coreset * 1000), cfg_draw.k,
                                 round(score_tmp)))
                cv2.imwrite(filename_out, img_figure[..., ::-1])
            else:
                writer.write(img_figure[..., ::-1])

            plt.clf()
            plt.close()

        if cfg_draw.mode_video:
            writer.release()


def assemble_patch(idx_patch, imgs_coreset, HW_map):
    size_receptive = imgs_coreset.shape[1]
    img_patch = np.zeros([(HW_map[0] * size_receptive),
                          (HW_map[1] * size_receptive), 3], dtype=np.uint8)

    # reset counter
    i_y = 0
    i_x = 0

    # loop of patch feature index
    for i_patch in idx_patch:
        # tile...
        img_piece = imgs_coreset[i_patch]
        y = i_y * size_receptive
        x = i_x * size_receptive
        img_patch[y:(y + size_receptive), x:(x + size_receptive)] = img_piece

        # count-up
        i_x += 1
        if i_x == HW_map[1]:
            i_x = 0
            i_y += 1

    return img_patch


def draw_curve(cfg_draw, x_img, y_img, auc_img, auc_img_mean,
                         x_pix, y_pix, auc_pix, auc_pix_mean, flg_roc=True):
    if flg_roc:
        idx = 'ROC'
        lbl_x = 'False Positive Rate'
        lbl_y = 'True Positive Rate'
    else:
        idx = 'PR'
        lbl_x = 'Recall'
        lbl_y = 'Precision'

    plt.figure(figsize=(15, 6), dpi=100, facecolor='white')
    for type_data in x_img.keys():
        plt.subplot(1, 2, 1)
        plt.plot(x_img[type_data], y_img[type_data],
                 label='%s %sAUC: %.3f' % (type_data, idx, auc_img[type_data]))
        plt.subplot(1, 2, 2)
        plt.plot(x_pix[type_data], y_pix[type_data],
                 label='%s %sAUC: %.3f' % (type_data, idx, auc_pix[type_data]))

    plt.subplot(1, 2, 1)
    plt.title('imagewise %sAUC %% : mean %.3f' % (idx, auc_img_mean))
    plt.grid()
    plt.legend(loc='lower right')
    plt.xlabel(lbl_x)
    plt.ylabel(lbl_y)
    plt.subplot(1, 2, 2)
    plt.title('pixelwise %sAUC %% : mean %.3f' % (idx, auc_pix_mean))
    plt.grid()
    plt.legend(loc='lower right')
    plt.xlabel(lbl_x)
    plt.ylabel(lbl_y)
    plt.gcf().tight_layout()
    plt.gcf().savefig('%s/%s-curve_p%04d_k%02d_aucimg%04d_aucpix%04d.png' %
                      (cfg_draw.path_result, idx.lower(),
                       (cfg_draw.percentage_coreset * 1000), cfg_draw.k,
                       round(auc_img_mean * 1000), round(auc_pix_mean * 1000)))
    plt.clf()
    plt.close()
