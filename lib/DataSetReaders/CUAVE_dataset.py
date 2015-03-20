import os
import scipy.io
import h5py
import hickle
import numpy

from theano import config

from MISC.container import ContainerRegisterMetaClass
from dataset_base import DatasetBase

TRAINING_PERCENT = 0.8

class CUAVEDataSet(DatasetBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(CUAVEDataSet, self).__init__(data_set_parameters)

    def build_dataset(self):

        #data_set = scipy.io.loadmat(self.dataset_path)
        data_set = hickle.load(file(self.dataset_path, 'r'))

        self.trainset = [numpy.astype(data_set['train_video'].T, config.floatX), numpy.astype(data_set['train_audio'].T,config.floatX)]
        self.testset = [numpy.astype(data_set['test_video'].T, config.floatX), numpy.astype(data_set['test_audio'].T, config.floatX)]
        self.tuning = self.testset