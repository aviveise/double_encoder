import os
import scipy.io
import h5py
import numpy

from theano.tensor import config, theano

from lib.MISC.container import ContainerRegisterMetaClass
from dataset_base import DatasetBase

TRAINING_PERCENT = 0.8


class GUYDataSet(DatasetBase):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        self._full = bool(int(data_set_parameters.get('full', 0)))
        super(GUYDataSet, self).__init__(data_set_parameters)
        self.reduce_test = 5

        if self._full:
            self.reduce_val = 5

        CNN_mat = h5py.File(os.path.join(self.dataset_path, 'visual_dataset_info.mat'), 'r')
        FV_mat = h5py.File(os.path.join(self.dataset_path, 'sentences_FV.mat'), 'r')
        map_mat = h5py.File(os.path.join(self.dataset_path, 'image_sent_map.mat'), 'r')
        idx_mat = h5py.File(os.path.join(self.dataset_path, 'dataset_info_idx.mat'), 'r')

        self._CNN_output = CNN_mat['image_vecs']
        self._feature_vectors = FV_mat['sent_vecs']
        self._images_sent_mapping = map_mat['image_idx_of_sent'].value

        self._training_sen_idx = idx_mat['trn_sent_I']
        self._training_image_idx = idx_mat['trn_images_I']

        self._validation_sen_idx = idx_mat['dev_sent_I']
        self._validation_image_idx = idx_mat['dev_images_I']

        self._test_sen_idx = idx_mat['tst_sent_I']
        self._test_image_idx = idx_mat['tst_images_I']

    def generate_mapping(self):

        self.x_y_mapping = {
            'test': self._get_mapping(self._test_sen_idx, self._images_sent_mapping),
            'dev': self._get_mapping(self._validation_sen_idx, self._images_sent_mapping)
        }

        self.x_reduce = {
            'dev': self._get_reduce_from_mapping(self.x_y_mapping['dev']),
            'test': self._get_reduce_from_mapping(self.x_y_mapping['test'])
        }

    def build_dataset(self):

        if self._full:
            train_size = self._training_sen_idx.shape[0]
            self.reduce_val = 5
        else:
            train_size = self._training_image_idx.shape[0]
            self.reduce_val = 0

        dev_size = self._validation_sen_idx.shape[0]
        test_size = self._test_sen_idx.shape[0]

        self.trainset = [numpy.ndarray((train_size, self._CNN_output.shape[1]), dtype=config.floatX),
                         numpy.ndarray((train_size, self._feature_vectors[0].shape[0]), dtype=config.floatX)]

        self.tuning = [numpy.ndarray((dev_size, self._CNN_output.shape[1]), dtype=config.floatX),
                       numpy.ndarray((dev_size, self._feature_vectors[0].shape[0]), dtype=config.floatX)]

        self.testset = [numpy.ndarray((test_size, self._CNN_output.shape[1]), dtype=config.floatX),
                        numpy.ndarray((test_size, self._feature_vectors[0].shape[0]), dtype=config.floatX)]

        self.x_y_mapping = {
            'train': self._get_mapping(self._training_sen_idx, self._images_sent_mapping),
            'test': self._get_mapping(self._test_sen_idx, self._images_sent_mapping),
            'dev': self._get_mapping(self._validation_sen_idx, self._images_sent_mapping)
        }

        for i in range(train_size):

            if self._full:
                self.trainset[0][i, :] = self._CNN_output[int(self._images_sent_mapping[int(self._training_sen_idx[i]) - 1]) - 1]
                self.trainset[1][i, :] = self._feature_vectors[int(self._training_sen_idx[i]) - 1]
            else:
                self.trainset[0][i, :] = self._CNN_output[int(self._training_image_idx[i]) - 1]
                self.trainset[1][i, :] = self._feature_vectors[
                    numpy.where(self._images_sent_mapping == self._training_image_idx[i])[0][0]]

        for i in range(dev_size):
            self.tuning[0][i, :] = self._CNN_output[int(self._images_sent_mapping[int(self._validation_sen_idx[i]) - 1]) - 1]
            self.tuning[1][i, :] = self._feature_vectors[int(self._validation_sen_idx[i]) - 1]
        for i in range(test_size):
            self.testset[0][i, :] = self._CNN_output[int(self._images_sent_mapping[int(self._test_sen_idx[i]) - 1]) - 1]
            self.testset[1][i, :] = self._feature_vectors[int(self._test_sen_idx[i]) - 1]


        self.generate_mapping()

    def __del__(self):
        del self._CNN_output
        del self._feature_vectors
        del self._images_sent_mapping

    def jitter(self, set):
        new_samples = set + numpy.cast[theano.config.floatX](numpy.random.normal(0, 0.25, set.shape))
        return new_samples

    def _get_mapping(self, sent_idx, images_sent_mapping):
        _mapping = None
        i = 1

        # Initializing state 0
        row = numpy.zeros(sent_idx.shape[0])
        row[0] = 1
        current_idx = images_sent_mapping[int(sent_idx[0]) - 1]

        while i < sent_idx.shape[0]:

            # If we are in a series we need to keep filling ones
            if images_sent_mapping[int(sent_idx[i]) - 1] == current_idx:
                row[i] = 1
            else:

                # Concatenate new row and start a new series
                if _mapping is None:
                    _mapping = row
                else:
                    _mapping = numpy.vstack((_mapping, row))
                row = numpy.zeros(sent_idx.shape[0])
                row[i] = 1
                current_idx = images_sent_mapping[int(sent_idx[i]) - 1]

            i += 1

        _mapping = numpy.vstack((_mapping, row))
        return _mapping

    def _get_reduce_from_mapping(self, mapping):
        _sum = numpy.sum(mapping, axis=1)
        _return = []
        for i in range(_sum.shape[0]):
            _return.append(int(numpy.sum(_sum[0: i])))

        return _return
