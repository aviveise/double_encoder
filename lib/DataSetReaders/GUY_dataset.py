import os
import scipy.io
import h5py
import numpy

from theano.tensor import config

from MISC.container import ContainerRegisterMetaClass
from dataset_base import DatasetBase

TRAINING_PERCENT = 0.8

class GUYDataSet(DatasetBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(GUYDataSet, self).__init__(data_set_parameters)

    def build_dataset(self):

        CNN_mat = h5py.File(os.path.join(self.dataset_path, 'visual_dataset_info.mat'), 'r')
        FV_mat = h5py.File(os.path.join(self.dataset_path, 'sentences_FV.mat'), 'r')
        idx_mat = h5py.File(os.path.join(self.dataset_path, 'dataset_info_idx.mat'), 'r')

        CNN_output = CNN_mat['image_vecs']
        feature_vectors = FV_mat['sent_vecs']
        images_sent_mapping = CNN_mat['image_idx_of_sent']

        training_image_idx = idx_mat['trn_images_I']
        training_sen_idx = idx_mat['trn_sent_I']

        validation_image_idx = idx_mat['dev_images_I']
        validation_sen_idx = idx_mat['dev_sent_I']

        test_image_idx = idx_mat['tst_images_I']
        test_sen_idx = idx_mat['tst_sent_I']

        self.trainset = [numpy.ndarray((CNN_output.shape[1], training_sen_idx.shape[0]), dtype=config.floatX),
                         numpy.ndarray((feature_vectors[0].shape[0], training_sen_idx.shape[0]), dtype=config.floatX)]

        self.tuning = [numpy.ndarray((CNN_output.shape[1], validation_sen_idx.shape[0]), dtype=config.floatX),
                       numpy.ndarray((feature_vectors[0].shape[0], validation_sen_idx.shape[0]), dtype=config.floatX)]

        self.testset = [numpy.ndarray((CNN_output.shape[1], test_sen_idx.shape[0]), dtype=config.floatX),
                        numpy.ndarray((feature_vectors[0].shape[0], test_sen_idx.shape[0]), dtype=config.floatX)]


        for i in range(min(training_sen_idx.shape[0], 50000)):

            self.trainset[0][:, i] = CNN_output[int(images_sent_mapping[int(training_sen_idx[i]) - 1]) - 1]
            self.trainset[1][:, i] = feature_vectors[int(training_sen_idx[i]) - 1]

        for i in range(validation_sen_idx.shape[0]):

            self.tuning[0][:, i] = CNN_output[int(images_sent_mapping[int(validation_sen_idx[i]) - 1]) - 1]
            self.tuning[1][:, i] = feature_vectors[int(validation_sen_idx[i]) - 1]

        for i in range(test_sen_idx.shape[0]):

            self.testset[0][:, i] = CNN_output[int(images_sent_mapping[int(test_sen_idx[i]) - 1]) - 1]
            self.testset[1][:, i] = feature_vectors[int(test_sen_idx[i]) - 1]
