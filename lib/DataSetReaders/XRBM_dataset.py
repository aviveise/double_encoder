import os
import struct
import random
import cPickle
import gzip

import numpy

from theano import config

from lib.MISC.container import ContainerRegisterMetaClass
from lib.DataSetReaders.dataset_base import DatasetBase

class XRBMDataSet(DatasetBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(XRBMDataSet, self).__init__(data_set_parameters)

    def build_dataset(self):

        train_x1_file_name = self.dataset_path + '/XRMB[JW11,numfr1=7,numfr2=7,fold0,training].dat'
        train_x2_file_name = self.dataset_path + '/MFCC[JW11,numfr1=7,numfr2=7,fold0,training].dat'

        self.trainset = (self.ReadBin(train_x1_file_name, 112), self.ReadBin(train_x2_file_name, 273))

        test_x1_file_name = self.dataset_path + '/XRMB[JW11,numfr1=7,numfr2=7,fold0,testing].dat'
        test_x2_file_name = self.dataset_path + '/MFCC[JW11,numfr1=7,numfr2=7,fold0,testing].dat'

        self.testset = (self.ReadBin(test_x1_file_name, 112), self.ReadBin(test_x2_file_name, 273))

        tuning_x1_file_name = self.dataset_path + '/XRMB[JW11,numfr1=7,numfr2=7,fold0,tuning].dat'
        tuning_x2_file_name = self.dataset_path + '/MFCC[JW11,numfr1=7,numfr2=7,fold0,tuning].dat'

        self.tuning = (self.ReadBin(tuning_x1_file_name, 112), self.ReadBin(tuning_x2_file_name, 273))

    def ReadBin(self, file_name, row_num, max_col_num=-1):

        f = open(file_name, 'rb')

        f.seek(0, os.SEEK_END)
        endPos = f.tell()
        f.seek(0, os.SEEK_SET)

        col_num = int(endPos / numpy.finfo(numpy.float32).nexp / row_num)

        if max_col_num > -1:
            col_num = max(col_num, max_col_num)

        set = numpy.ndarray([row_num, col_num], dtype=config.floatX)

        for i in xrange(col_num):
            for j in xrange(row_num):
                set[j, i] = struct.unpack('d', f.read(8))[0]

        f.close()

        return set

class XRBMDataSetOld(object):

    def __init__(self, file_path):

        print '--------------Loading XRBM Data----------------'

        f = gzip.open(file_path, 'rb')
        train_set, test_set = cPickle.load(f)
        f.close()

        self.trainset = train_set[0].astype(config.floatX, copy=False), train_set[1].astype(config.floatX, copy=False)
        self.testset = test_set[0].astype(config.floatX, copy=False), test_set[1].astype(config.floatX, copy=False)

        x1_train_set, x1_tuning_set, test_samples = self.produce_optimization_sets(self.trainset[0])
        x2_train_set, x2_tuning_set, test_samples = self.produce_optimization_sets(self.trainset[1], test_samples)

        self.trainset = x1_train_set, x2_train_set
        self.tuning = x1_tuning_set, x2_tuning_set

        print 'Training set size = %d' % self.trainset[0].shape[1]
        print 'Test set size = %d' % self.testset[0].shape[1]
        print 'Tuning set size = %d' % self.tuning[0].shape[1]

    def produce_optimization_sets(self, train, test_samples=None):

        test_size = int(round(train.shape[1] / 10))

        if test_samples is None:
            test_samples = random.sample(xrange(0, train.shape[1] - 1), test_size)

        train_index = 0
        test_index = 0

        train_result = numpy.ndarray([train.shape[0], train.shape[1] - test_size], dtype=config.floatX)
        test_result = numpy.ndarray([train.shape[0], test_size], dtype=config.floatX)

        for i in xrange(train.shape[1]):

            if i in test_samples:
                test_result[:, test_index] = train[:, i]
                test_index += 1
            else:
                train_result[:, train_index] = train[:, i]
                train_index += 1

        return [train_result, test_result, test_samples]
