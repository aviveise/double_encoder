import os
import sys

import numpy
import cv2
import gzip
import cPickle

from theano import config
from MISC.container import ContainerRegisterMetaClass
from DataSetReaders.dataset_base import DatasetBase

TRAINING_PERCENT = 0.8

class JAPANDataSetVideo(DatasetBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(JAPANDataSetVideo, self).__init__(data_set_parameters)

    def build_dataset(self):

        cap = cv2.VideoCapture(self.dataset_path)

        LT = numpy.ndarray([1200, 390], dtype=config.floatX)
        RB = numpy.ndarray([1200, 390], dtype=config.floatX)

        for i in xrange(390):
            for j in xrange(9):
                rat, frame = cap.read()

            rat, frame = cap.read()
            frame = self.process_frame(frame, [960, 1280])

            LT[:, i] = frame[:, 0]
            RB[:, i] = frame[:, 3]

        training_size = int(LT.shape[1] * TRAINING_PERCENT * 0.9)
        tuning_size = int(LT.shape[1] * TRAINING_PERCENT * 0.1)
        test_size = int(LT.shape[1] * (1 - TRAINING_PERCENT))

        self.trainset = [LT[:, 0: training_size].astype(config.floatX, copy=False),
                         RB[:, 0: training_size].astype(config.floatX, copy=False)]

        self.tuning = [LT[:, training_size: training_size + tuning_size].astype(config.floatX, copy=False),
                       RB[:, training_size: training_size + tuning_size].astype(config.floatX, copy=False)]

        self.testset = [LT[:, training_size + tuning_size: training_size + tuning_size + test_size].astype(config.floatX, copy=False),
                       RB[:, training_size + tuning_size: training_size + tuning_size + test_size].astype(config.floatX, copy=False)]

        return


    def process_frame(self, frame, size):

        szr = size[0] / 32
        szc = size[1] / 32

        result = numpy.ndarray([szr * szc, 4], dtype=config.floatX)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (image.shape[0]/16, image.shape[1]/16), 0, 0, cv2.INTER_LINEAR)

        result[:, 0] = image[0: szc, 0: szr].reshape([1200])
        result[:, 1] = image[szc: (szc*2), 0: szr].reshape([1200])
        result[:, 2] = image[0: szc, szr: szr*2].reshape([1200])
        result[:, 3] = image[szc: szc*2, szr: szr*2].reshape([1200])

        return result

class JAPANDataSet(DatasetBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, dataset_path, center=False, normalize=False, whiten=False):
        super(JAPANDataSet, self).__init__(dataset_path, 'JAPAN', center, normalize, whiten)

    def build_dataset(self):

        f = gzip.open(self.dataset_path, 'rb')
        trainset, tuning, testset = cPickle.load(f)
        f.close()

        self.trainset = trainset[0].astype(config.floatX, copy=False), \
                        trainset[1].astype(config.floatX, copy=False)

        self.tuning = tuning[0].astype(config.floatX, copy=False), \
                       tuning[1].astype(config.floatX, copy=False)

        self.testset = testset[0].astype(config.floatX, copy=False), \
                       testset[1].astype(config.floatX, copy=False)