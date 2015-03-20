import os
import scipy.io
import h5py

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
        data_set = h5py.File(self.dataset_path, 'r')

        self.trainset = [data_set['train_video'], data_set['train_audio']]
        self.testset = [data_set['test_video'], data_set['test_audio']]