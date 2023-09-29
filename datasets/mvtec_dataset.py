import os
import numpy as np
from utils.config import ConfigData
from datasets.shared_memory import SharedMemory


class MVTecDataset:
    @classmethod
    def __init__(cls, type_data):
        cls.type_data = type_data
        cls.SHAPE_INPUT = ConfigData.SHAPE_INPUT

        cls.imgs_train = None
        cls.imgs_test = {}
        cls.gts_test = {}

        # read train data
        desc = 'read images for train (case:good)'
        path = os.path.join(ConfigData.path_parent, type_data, 'train/good')
        files = [os.path.join(path, f) for f in os.listdir(path)
                 if (os.path.isfile(os.path.join(path, f)) & ('.png' in f))]
        cls.files_train = np.sort(np.array(files))
        cls.imgs_train = SharedMemory.read_img_parallel(files=cls.files_train,
                                                        imgs=cls.imgs_train,
                                                        desc=desc)
        if ConfigData.shuffle:
            np.random.seed(ConfigData.seed)
            cls.imgs_train = np.random.permutation(cls.imgs_train)
        if ConfigData.flip_horz:
            cls.imgs_train = np.concatenate([cls.imgs_train,
                                             cls.imgs_train[:, :, ::-1]], axis=0)
        if ConfigData.flip_vert:
            cls.imgs_train = np.concatenate([cls.imgs_train,
                                             cls.imgs_train[:, ::-1]], axis=0)

        # read test data
        cls.files_test = {}
        cls.types_test = os.listdir(os.path.join(ConfigData.path_parent, type_data, 'test'))
        cls.types_test = np.array(sorted(cls.types_test))
        for type_test in cls.types_test:
            desc = 'read images for test (case:%s)' % type_test
            path = os.path.join(ConfigData.path_parent, type_data, 'test', type_test)
            files = [os.path.join(path, f) for f in os.listdir(path)
                     if (os.path.isfile(os.path.join(path, f)) & ('.png' in f))]
            cls.files_test[type_test] = np.sort(np.array(files))
            cls.imgs_test[type_test] = None
            cls.imgs_test[type_test] = SharedMemory.read_img_parallel(files=cls.files_test[type_test],
                                                                      imgs=cls.imgs_test[type_test],
                                                                      desc=desc)

        # read ground truth of test data
        for type_test in cls.types_test:
            # create memory shared variable
            if type_test == 'good':
                cls.gts_test[type_test] = np.zeros([len(cls.files_test[type_test]),
                                                    ConfigData.SHAPE_INPUT[0],
                                                    ConfigData.SHAPE_INPUT[1]], dtype=np.uint8)
            else:
                desc = 'read ground-truths for test (case:%s)' % type_test
                cls.gts_test[type_test] = None
                cls.gts_test[type_test] = SharedMemory.read_img_parallel(files=cls.files_test[type_test],
                                                                         imgs=cls.gts_test[type_test],
                                                                         is_gt=True,
                                                                         desc=desc)


