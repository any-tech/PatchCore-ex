import os
import numpy as np
import cv2
from tqdm import tqdm
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
        path = '%s/%s/train/good' % (ConfigData.path_data, type_data)
        files = ['%s/%s' % (path, f) for f in os.listdir(path)
                 if (os.path.isfile('%s/%s' % (path, f)) & (('.png' in f) | ('.jpg' in f)))]
        cls.files_train = np.sort(np.array(files))
        cls.imgs_train = SharedMemory.read_img_parallel(files=cls.files_train,
                                                        imgs=cls.imgs_train,
                                                        desc=desc)
        if ConfigData.shuffle:
            cls.imgs_train = np.random.permutation(cls.imgs_train)
        if ConfigData.flip_horz:
            cls.imgs_train = np.concatenate([cls.imgs_train,
                                             cls.imgs_train[:, :, ::-1]], axis=0)
        if ConfigData.flip_vert:
            cls.imgs_train = np.concatenate([cls.imgs_train,
                                             cls.imgs_train[:, ::-1]], axis=0)

        # read test data
        cls.files_test = {}
        cls.types_test = os.listdir('%s/%s/test' % (ConfigData.path_data, type_data))
        cls.types_test = np.array(sorted(cls.types_test))
        for type_test in cls.types_test:
            desc = 'read images for test (case:%s)' % type_test
            path = '%s/%s/test/%s' % (ConfigData.path_data, type_data, type_test)
            files = [('%s/%s' % (path, f)) for f in os.listdir(path)
                     if (os.path.isfile('%s/%s' % (path, f)) & (('.png' in f) | ('.jpg' in f)))]
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


class MVTecDatasetOnlyTest:
    @classmethod
    def __init__(cls, type_data):
        if ConfigData.path_data is not None:
            cls.type_data = type_data
            cls.SHAPE_INPUT = ConfigData.SHAPE_INPUT

            # read test data
            cls.imgs_test = {}
            cls.files_test = {}
            cls.types_test = os.listdir('%s/%s/test' % (ConfigData.path_data, type_data))
            cls.types_test = np.array(sorted(cls.types_test))
            for type_test in cls.types_test:
                desc = 'read images for test (case:%s)' % type_test
                path = '%s/%s/test/%s' % (ConfigData.path_data, type_data, type_test)
                files = [('%s/%s' % (path, f)) for f in os.listdir(path)
                         if (os.path.isfile('%s/%s' % (path, f)) & (('.png' in f) | ('.jpg' in f)))]
                cls.files_test[type_test] = np.sort(np.array(files))
                cls.imgs_test[type_test] = None
                cls.imgs_test[type_test] = SharedMemory.read_img_parallel(files=cls.files_test[type_test],
                                                                          imgs=cls.imgs_test[type_test],
                                                                          desc=desc)
        else:
            cls.type_data = type_data
            cls.SHAPE_INPUT = ConfigData.SHAPE_INPUT

            # read test data
            cls.imgs_test = {}
            cls.files_test = {}
            cls.types_test = np.array(['video'])
            type_test = cls.types_test[0]
            cls.imgs_test[type_test] = []
            cls.files_test[type_test] = []
            capture = cv2.VideoCapture(ConfigData.path_video)
            num_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            desc = 'read images from video for test'
            for i_frame in tqdm(range(num_frame), desc=desc):
                # frame read
                capture.grab()
                ret, frame = capture.retrieve()
                if ret:
                    frame = frame[..., ::-1]  # BGR2RGB
                    frame = cv2.resize(frame, (ConfigData.SHAPE_MIDDLE[1],
                                               ConfigData.SHAPE_MIDDLE[0]),
                                       interpolation=cv2.INTER_AREA)
                    frame = frame[ConfigData.pixel_cut[0]:(ConfigData.SHAPE_INPUT[0] +
                                                           ConfigData.pixel_cut[0]),
                                  ConfigData.pixel_cut[1]:(ConfigData.SHAPE_INPUT[1] +
                                                           ConfigData.pixel_cut[1])]
                    cls.imgs_test[type_test].append(frame)
                    cls.files_test[type_test].append('%s/frame/%05d' %
                                                     (os.path.basename(ConfigData.path_video),
                                                      i_frame))
            cls.imgs_test[type_test] = np.array(cls.imgs_test[type_test])
            cls.files_test[type_test] = np.array(cls.files_test[type_test])
