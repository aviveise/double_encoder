import abc
import os
import struct

import random
import hickle
import scipy
import theano
import numpy
import pickle

from sklearn.decomposition import PCA
from MISC.utils import scale_cols
from MISC.utils import center as center_function
from MISC.utils import normalize as normalize_function
from MISC.whiten_transform import WhitenTransform
from MISC.logger import OutputLog

__author__ = 'aviv'


class DatasetBase(object):
    def __init__(self, data_set_parameters):

        OutputLog().write('Loading dataset: ' + data_set_parameters['name'])

        self.dataset_path = data_set_parameters['path']
        self.trainset = None
        self.testset = None
        self.tuning = None
        self.negatives = None
        self.reduce_test = 0
        self.reduce_val = 0

        scale = bool(int(data_set_parameters['scale']))
        scale_rows = bool(int(data_set_parameters['scale_samples']))
        whiten = bool(int(data_set_parameters['whiten']))
        pca = map(int, data_set_parameters['pca'].split())

        path = self.dataset_path
        if not os.path.isdir(self.dataset_path):
            path = os.path.dirname(os.path.abspath(self.dataset_path))

        train_file = os.path.join(path, 'train.p')
        test_file = os.path.join(path, 'test.p')
        validate_file = os.path.join(path, 'validate.p')

        if os.path.exists(train_file) and \
                os.path.exists(test_file) and \
                os.path.exists(validate_file):
            self.trainset = hickle.load(file(train_file, 'r'))
            self.testset = hickle.load(file(test_file, 'r'))
            self.tuning = hickle.load(file(validate_file, 'r'))

            self.trainset = self.trainset[0].astype(dtype=theano.config.floatX), self.trainset[1].astype(
                dtype=theano.config.floatX)
            self.testset = self.testset[0].astype(dtype=theano.config.floatX), self.testset[1].astype(
                dtype=theano.config.floatX)
            self.tuning = self.tuning[0].astype(dtype=theano.config.floatX), self.tuning[1].astype(
                dtype=theano.config.floatX)

            OutputLog().write('Dataset dimensions = %d, %d' % (self.trainset[0].shape[1], self.trainset[1].shape[1]))
            OutputLog().write('Training set size = %d' % self.trainset[0].shape[0])
            OutputLog().write('Test set size = %d' % self.testset[0].shape[0])

            return

        self.build_dataset()

        if scale_rows:
            self.trainset[0] = scale_cols(self.trainset[0].T)[0].T
            self.trainset[1] = scale_cols(self.trainset[1].T)[0].T

            self.tuning[0] = scale_cols(self.tuning[0].T)[0].T
            self.tuning[1] = scale_cols(self.tuning[1].T)[0].T

            self.testset[0] = scale_cols(self.testset[0].T)[0].T
            self.testset[1] = scale_cols(self.testset[1].T)[0].T

        if scale:
            train_set_x, scaler_x = scale_cols(self.trainset[0])
            train_set_y, scaler_y = scale_cols(self.trainset[1])

            self.trainset = train_set_x, train_set_y
            self.tuning = scaler_x.transform(self.tuning[0]), scaler_y.transform(self.tuning[1])
            self.testset = scaler_x.transform(self.testset[0]), scaler_y.transform(self.testset[1])

            scaler_x_path = os.path.join(path, 'scaler_x.p')
            scaler_y_path = os.path.join(path, 'scaler_y.p')

            pickle.dump(scaler_x, file(scaler_x_path, 'w'))
            pickle.dump(scaler_y, file(scaler_y_path, 'w'))

        if not pca[0] == 0 and not pca[1] == 0:
            pca_dim1 = PCA(pca[0], whiten)
            pca_dim2 = PCA(pca[1], whiten)

            pca_dim1.fit(self.trainset[0])
            pca_dim2.fit(self.trainset[1])

            self.trainset = (pca_dim1.transform(self.trainset[0]), pca_dim2.transform(self.trainset[1]))
            self.testset = (pca_dim1.transform(self.testset[0]), pca_dim2.transform(self.testset[1]))
            self.tuning = (pca_dim1.transform(self.tuning[0]), pca_dim2.transform(self.tuning[1]))

        if whiten:
            OutputLog().write('using whiten')
            pca_dim1 = PCA(whiten=True)
            pca_dim2 = PCA(whiten=True)

            pca_dim1.fit(self.trainset[0])
            pca_dim2.fit(self.trainset[1])

            self.trainset = (pca_dim1.transform(self.trainset[0]), pca_dim2.transform(self.trainset[1]))
            self.testset = (pca_dim1.transform(self.testset[0]), pca_dim2.transform(self.testset[1]))
            self.tuning = (pca_dim1.transform(self.tuning[0]), pca_dim2.transform(self.tuning[1]))

        hickle.dump(self.trainset, file(train_file, 'w'))
        hickle.dump(self.testset, file(test_file, 'w'))
        hickle.dump(self.tuning, file(validate_file, 'w'))

        OutputLog().write('Dataset dimensions = %d, %d' % (self.trainset[0].shape[1], self.trainset[1].shape[1]))
        OutputLog().write('Training set size = %d' % self.trainset[0].shape[0])
        OutputLog().write('Test set size = %d' % self.testset[0].shape[0])

        if self.tuning is not None:
            OutputLog().write('Tuning set size = %d' % self.tuning[0].shape[0])

    def produce_optimization_sets(self, train, test_samples=None):

        if test_samples == 0:
            return [train, numpy.ndarray([0, 0]), 0]

        test_size = int(round(train.shape[0] / 10))

        if test_samples is None:
            test_samples = random.sample(xrange(0, train.shape[0] - 1), test_size)

        train_index = 0
        test_index = 0

        train_result = numpy.ndarray([train.shape[0] - test_size, train.shape[1]], dtype=theano.config.floatX)
        test_result = numpy.ndarray([test_size, train.shape[1]], dtype=theano.config.floatX)

        for i in xrange(train.shape[0]):

            if i in test_samples:
                test_result[test_index, :] = train[i, :]
                test_index += 1
            else:
                train_result[train_index, :] = train[i, :]
                train_index += 1

        return [train_result, test_result, test_samples]

    @abc.abstractmethod
    def build_dataset(self):
        """main optimization method"""
        return

    def export_dat(self):

        training0_dat_name = self.name + '_training_0.dat'
        training1_dat_name = self.name + '_training_1.dat'

        tuning0_dat_name = self.name + '_tuning_0.dat'
        tuning1_dat_name = self.name + '_tuning_1.dat'

        test0_dat_name = self.name + '_test_0.dat'
        test1_dat_name = self.name + '_test_1.dat'

        training0_dat_file = open(training0_dat_name, 'w+')
        self.write_dat(self.trainset[0], training0_dat_file)
        training0_dat_file.close()

        training1_dat_file = open(training1_dat_name, 'w+')
        self.write_dat(self.trainset[1], training1_dat_file)
        training1_dat_file.close()

        tuning0_dat_file = open(tuning0_dat_name, 'w+')
        self.write_dat(self.tuning[0], tuning0_dat_file)
        tuning0_dat_file.close()

        tuning1_dat_file = open(tuning1_dat_name, 'w+')
        self.write_dat(self.tuning[1], tuning1_dat_file)
        tuning1_dat_file.close()

        test0_dat_file = open(test0_dat_name, 'w+')
        self.write_dat(self.testset[0], test0_dat_file)
        test0_dat_file.close()

        test1_dat_file = open(test1_dat_name, 'w+')
        self.write_dat(self.testset[1], test1_dat_file)
        test1_dat_file.close()

    def write_dat(self, dataset, dat_file):

        for i in xrange(dataset.shape[1]):
            for j in xrange(dataset.shape[0]):
                dat_file.write(struct.pack('d', dataset[j, i]))

    def _generate_negatives(self, x, y):

        complete_shuffle = False
        sample_num = x.shape[0]

        while not complete_shuffle:
            shuffle_index = numpy.random.permutation(sample_num)
            if numpy.sum(shuffle_index == range(sample_num)) == 0:
                complete_shuffle = True

        return x[shuffle_index], y
