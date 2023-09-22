import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
from utils.config import ConfigData


def read_and_resize(file):
    img = cv2.imread(file)[..., ::-1]  # BGR2RGB
    img = cv2.resize(img, (ConfigData.SHAPE_MIDDLE[1], ConfigData.SHAPE_MIDDLE[0]),
                     interpolation=cv2.INTER_AREA)
    img = img[ConfigData.pixel_cut[0]:(ConfigData.SHAPE_INPUT[0] + ConfigData.pixel_cut[0]),
              ConfigData.pixel_cut[1]:(ConfigData.SHAPE_INPUT[1] + ConfigData.pixel_cut[1])]
    SharedMemory.shared_array[np.where(SharedMemory.files == file)[0]] = img


def read_and_resize_ground_truth(file):
    file_gt = file.replace('/test/', '/ground_truth/')
    file_gt = file_gt.replace('.png', '_mask.png')

    gt = cv2.imread(file_gt, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, (ConfigData.SHAPE_MIDDLE[1], ConfigData.SHAPE_MIDDLE[0]),
                    interpolation=cv2.INTER_NEAREST)
    gt = gt[ConfigData.pixel_cut[0]:(ConfigData.SHAPE_INPUT[0] + ConfigData.pixel_cut[0]),
            ConfigData.pixel_cut[1]:(ConfigData.SHAPE_INPUT[1] + ConfigData.pixel_cut[1])]

    if np.max(gt) != 0:
        gt = (gt / np.max(gt)).astype(np.uint8)

    SharedMemory.shared_array[np.where(SharedMemory.files == file)[0]] = gt


class SharedMemory:
    @classmethod
    def read_img_parallel(cls, files, imgs, is_gt=False, desc='read images'):
        cls.files = files
        cls.shared_array = imgs

        if not is_gt:
            shape = (len(cls.files), ConfigData.SHAPE_INPUT[0], ConfigData.SHAPE_INPUT[1], 3)
            num_elm = shape[0] * shape[1] * shape[2] * shape[3]
        else:
            shape = (len(cls.files), ConfigData.SHAPE_INPUT[0], ConfigData.SHAPE_INPUT[1])
            num_elm = shape[0] * shape[1] * shape[2]

        ctype = np.ctypeslib.as_ctypes_type(np.dtype(np.uint8))
        data = np.ctypeslib.as_array(RawArray(ctype, num_elm))
        data.shape = shape
        cls.shared_array = data.view(np.uint8)

        # exec imread and imresize on multiprocess
        mp.set_start_method('fork', force=True)
        p = mp.Pool(min(mp.cpu_count(), ConfigData.num_cpu_max))

        func = read_and_resize
        if is_gt:
            func = read_and_resize_ground_truth

        for _ in tqdm(p.imap_unordered(func, cls.files), total=len(cls.files), desc=desc):
            pass

        p.close()

        return cls.shared_array
