import abc
import os
import struct

import random

import cPickle

import functools
import hickle
import theano
import numpy
import pickle

from sklearn import preprocessing
from sklearn.decomposition import PCA
from lib.MISC.utils import scale_cols, normalize
from lib.MISC.logger import OutputLog

__author__ = 'aviv'


class IdentityPreprocessor():
    def transform(self, x):
        return x


class TransposedPreprocessor():
    def __init__(self, preprocessor):
        self._preprocessor = preprocessor

    def transform(self, x):
        return self._preprocessor.transform(x.T).T

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
        self.x_y_mapping = {'train': None, 'dev': None, 'test': None}
        self.x_reduce = {'train': None, 'dev': None, 'test': None}

        self.data_set_parameters = data_set_parameters
        self.scale = bool(int(data_set_parameters['scale']))
        self.scale_rows = bool(int(data_set_parameters['scale_samples']))
        self.whiten = bool(int(data_set_parameters['whiten']))
        self.pca = map(int, data_set_parameters['pca'].split())
        self.normalize_data = bool(int(data_set_parameters['normalize']))

    def load(self):
        path = self.dataset_path
        if not os.path.isdir(self.dataset_path):
            path = os.path.dirname(os.path.abspath(self.dataset_path))

        params = os.path.join(path, 'params.p')

        try:
            self.trainset = self.load_cache(path, 'train')
            self.testset = self.load_cache(path, 'test')
            self.tuning = self.load_cache(path, 'validate')

            try:
	    	self.x_y_mapping['train'] = numpy.load(os.path.join(path, 'mapping_train.npy'), 'r')
            except:
	        OutputLog().write('Failed loading training mapping')

	    self.x_y_mapping['test'] = numpy.load(os.path.join(path, 'mapping_test.npy'), 'r')
            self.x_y_mapping['dev'] = numpy.load(os.path.join(path, 'mapping_dev.npy'), 'r')

            self.x_reduce = cPickle.load(open(os.path.join(path, 'reduce.p'), 'r'))

            with open(params) as params_file:
                loaded_params = cPickle.load(params_file)

            OutputLog().write('Loaded dataset params: {0}'.format(loaded_params))

        except Exception as e:
            OutputLog().write('Failed loading from local cache with exception: {}'.format(e))
            self.build_dataset()

        self.preprocess()

        OutputLog().write('Dataset dimensions = %d, %d' % (self.trainset[0].shape[1], self.trainset[1].shape[1]))
        OutputLog().write('Training set size = %d' % self.trainset[0].shape[0])
        OutputLog().write('Test set size = %d' % self.testset[0].shape[0])

        OutputLog().write('Dataset params: {0}'.format(self.data_set_parameters))

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

    def preprocess(self, copy=False):

        path = self.dataset_path
        if not os.path.isdir(self.dataset_path):
            path = os.path.dirname(os.path.abspath(self.dataset_path))

        if self.normalize_data:
            self.preprocessors = (functools.partial(preprocessing.normalize, copy=copy),
                                  functools.partial(preprocessing.normalize, copy=copy))

        if self.scale:
            self.preprocessors = (preprocessing.StandardScaler(copy=copy).fit(self.trainset[0]).transform,
                                  preprocessing.StandardScaler(copy=copy).fit(self.trainset[1]).transform)

        if self.scale_rows:
            self.preprocessors = (functools.partial(preprocessing.scale, copy=copy, axis=1),
                                  functools.partial(preprocessing.scale, copy=copy, axis=1))

        if not self.pca[0] == 0:
            self.preprocessors = (PCA(self.pca[0], copy=copy, whiten=self.whiten).fit(self.trainset[0]).transform,
                                  lambda x: x)

        if not self.pca[1] == 0:
            self.preprocessors = (lambda x: x,
                                  PCA(self.pca[0], copy=copy, whiten=self.whiten).fit(self.trainset[1]).transform)

        if self.whiten:
            OutputLog().write('using whiten')
            pca_dim1 = PCA(whiten=True)
            pca_dim2 = PCA(whiten=True)

            pca_dim1.fit(self.trainset[0])
            pca_dim2.fit(self.trainset[1])

            self.trainset = (pca_dim1.transform(self.trainset[0]), pca_dim2.transform(self.trainset[1]))
            self.testset = (pca_dim1.transform(self.testset[0]), pca_dim2.transform(self.testset[1]))
            self.tuning = (pca_dim1.transform(self.tuning[0]), pca_dim2.transform(self.tuning[1]))

    def dump(self, suffix=''):

        path = self.dataset_path
        if not os.path.isdir(self.dataset_path):
            path = os.path.dirname(os.path.abspath(self.dataset_path))

        train_file = os.path.join(path, 'train{0}.p'.format(suffix))
        test_file = os.path.join(path, 'test{0}.p'.format(suffix))
        validate_file = os.path.join(path, 'validate{0}.p'.format(suffix))
        params_file = os.path.join(path, 'params{0}.p'.format(suffix))
        mapping_file = os.path.join(path, 'mapping{0}.p'.format(suffix))

        numpy.save('x_{0}'.format(train_file), self.trainset[0])
        numpy.save('y_{0}'.format(train_file), self.trainset[1])

        # hickle.dump(self.trainset, file(train_file, 'w'))
        hickle.dump(self.testset, file(test_file, 'w'))
        hickle.dump(self.tuning, file(validate_file, 'w'))
        hickle.dump(self.data_set_parameters, file(params_file, 'w'))
        hickle.dump({'map': self.x_y_mapping, 'reduce': self.x_reduce}, file(mapping_file, 'w'))

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

    def load_cache(self, path, type, mmap_type='r'):
        dataset_x = numpy.load(os.path.join(path, type + '_x.npy'), mmap_type)
        dataset_y = numpy.load(os.path.join(path, type + '_y.npy'), mmap_type)

        return (dataset_x, dataset_y)
