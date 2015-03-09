import abc
import struct

import random
import theano
import numpy

from MISC.utils import center as center_function, unitnorm
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

        self.build_dataset()

        if center:
            center_function(self.trainset[0])
            center_function(self.tuning[0])
            center_function(self.testset[0])
            center_function(self.trainset[1])
            center_function(self.tuning[1])
            center_function(self.testset[1])

        if normalize:
            unitnorm(self.trainset[0])
            unitnorm(self.tuning[0])
            unitnorm(self.testset[0])
            unitnorm(self.trainset[1])
            unitnorm(self.tuning[1])
            unitnorm(self.testset[1])

        if whiten:
            transform_0 = WhitenTransform(self.trainset[0])
            transform_1 = WhitenTransform(self.trainset[1])
            self.trainset = (transform_0.transform(self.trainset[0]), transform_1.transform(self.trainset[1]))
            self.testset = (transform_0.transform(self.testset[0]), transform_1.transform(self.testset[1]))
            self.tuning = (transform_0.transform(self.tuning[0]), transform_1.transform(self.tuning[1]))

        OutputLog().write('Training set size = %d' % self.trainset[0].shape[1])
        OutputLog().write('Test set size = %d' % self.testset[0].shape[1])
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
