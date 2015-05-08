import abc
import os
import struct

import random
import hickle
import scipy
import theano
import numpy

from sklearn.decomposition import PCA
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
        
        normalize = bool(int(data_set_parameters['normalize']))
        center = bool(int(data_set_parameters['center']))
        whiten = bool(int(data_set_parameters['whiten']))
        pca = map(int, data_set_parameters['pca'].split())

        path = self.dataset_path
        if not os.path.isdir(self.dataset_path):
            path = os.path.dirname(os.path.abspath(self.dataset_path))

        train_file = os.path.join(path, 'train.p')
        test_file = os.path.join(path, 'test.p')
        validate_file = os.path.join(path, 'validate.p')

        if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(validate_file):
            self.trainset = hickle.load(file(train_file, 'r'))
            self.testset = hickle.load(file(test_file, 'r'))
            self.tuning = hickle.load(file(validate_file, 'r'))
            return

        self.build_dataset()

        if center:
            train_set_x, mean_x = center_function(self.trainset[0])
            train_set_y, mean_y = center_function(self.trainset[1])

            self.trainset = train_set_x, train_set_y
            self.tuning = self.tuning[0] - mean_x * numpy.ones([1, self.tuning[0].shape[1]], dtype=theano.config.floatX), \
                          self.tuning[1] - mean_y * numpy.ones([1, self.tuning[1].shape[1]], dtype=theano.config.floatX)

            self.testset = self.testset[0] - mean_x * numpy.ones([1, self.testset[0].shape[1]], dtype=theano.config.floatX),\
                           self.testset[1] - mean_y * numpy.ones([1, self.testset[1].shape[1]], dtype=theano.config.floatX)
        if normalize:
            train_set_x, norm_x = normalize_function(self.trainset[0])
            train_set_y, norm_y = normalize_function(self.trainset[1])

            self.trainset = train_set_x, train_set_y
            self.tuning = self.tuning[0] / norm_x * numpy.ones([1, self.tuning[0].shape[1]], dtype=theano.config.floatX), \
                          self.tuning[1] / norm_y * numpy.ones([1, self.tuning[1].shape[1]], dtype=theano.config.floatX)

            self.testset = self.testset[0] / norm_x * numpy.ones([1, self.testset[0].shape[1]], dtype=theano.config.floatX),\
                           self.testset[1] / norm_y * numpy.ones([1, self.testset[1].shape[1]], dtype=theano.config.floatX)
        if not pca[0] == 0 and not pca[1] == 0:

            pca_dim1 = PCA(pca[0], whiten)
            pca_dim2 = PCA(pca[1], whiten)

            pca_dim1.fit(self.trainset[0].T)
            pca_dim2.fit(self.trainset[1].T)

            self.trainset = (pca_dim1.transform(self.trainset[0].T).T, pca_dim2.transform(self.trainset[1].T).T)
            self.testset = (pca_dim1.transform(self.testset[0].T).T, pca_dim2.transform(self.testset[1].T).T)
            self.tuning = (pca_dim1.transform(self.tuning[0].T).T, pca_dim2.transform(self.tuning[1].T).T)

        if whiten:
            print 'using whiten'
            whiten_image_path = os.path.join(self.dataset_path, 'whiten_image.mat')
            whiten_sen_path = os.path.join(self.dataset_path, 'whiten_sen.mat')

            if not os.path.isdir(self.dataset_path):
                dir = os.path.split(self.dataset_path)[0]
                whiten_image_path = os.path.join(dir, 'whiten_image.mat')
                whiten_sen_path = os.path.join(dir, 'whiten_sen.mat')

            if os.path.exists(whiten_image_path) and os.path.exists(whiten_sen_path):
                print 'loading whiten matrices from files'
                wx = scipy.io.loadmat(whiten_image_path)['w'].astype(dtype=theano.config.floatX)
                wy = scipy.io.loadmat(whiten_sen_path)['w'].astype(dtype=theano.config.floatX)
            else:
                print 'cache not found calculating whiten transforms'
                wx = WhitenTransform.fit(self.trainset[0]).astype(dtype=theano.config.floatX)
                wy = WhitenTransform.fit(self.trainset[1]).astype(dtype=theano.config.floatX)

                print 'saving matrices of sizes {0}, {1}'.format(wx.shape, wy.shape)
                scipy.io.savemat(whiten_image_path, {'w': wx})
                scipy.io.savemat(whiten_sen_path, {'w': wy})

            print 'transforming data'
            self.trainset = (WhitenTransform.transform(self.trainset[0], wx),
                             WhitenTransform.transform(self.trainset[1], wy))

            self.testset = (WhitenTransform.transform(self.testset[0], wx),
                            WhitenTransform.transform(self.testset[1], wy))

            self.tuning = (WhitenTransform.transform(self.tuning[0], wx),
                           WhitenTransform.transform(self.tuning[1], wy))

        hickle.dump(self.trainset, file(train_file, 'w'))
        hickle.dump(self.testset, file(test_file, 'w'))
        hickle.dump(self.tuning, file(validate_file, 'w'))

        OutputLog().write('Dataset dimensions = %d, %d' % (self.trainset[0].shape[0], self.trainset[1].shape[0]))
        OutputLog().write('Training set size = %d' % self.trainset[0].shape[1])
        OutputLog().write('Test set size = %d' % self.testset[0].shape[1])

        if self.tuning is not None:
            OutputLog().write('Tuning set size = %d' % self.tuning[0].shape[1])

    def produce_optimization_sets(self, train, test_samples=None):

        if test_samples == 0:
            return [train, numpy.ndarray([0, 0]), 0]

        test_size = int(round(train.shape[1] / 10))

        if test_samples is None:
            test_samples = random.sample(xrange(0, train.shape[1] - 1), test_size)

        train_index = 0
        test_index = 0

        train_result = numpy.ndarray([train.shape[0], train.shape[1] - test_size], dtype=theano.config.floatX)
        test_result = numpy.ndarray([train.shape[0], test_size], dtype=theano.config.floatX)

        for i in xrange(train.shape[1]):

            if i in test_samples:
                test_result[:, test_index] = train[:, i]
                test_index += 1
            else:
                train_result[:, train_index] = train[:, i]
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
