import os
import struct
import random
import cPickle
import gzip
import theano

import numpy

from theano import config

from lib.MISC.container import ContainerRegisterMetaClass
from dataset_base import DatasetBase


class XRBMDataSet(DatasetBase):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(XRBMDataSet, self).__init__(data_set_parameters)

    def build_dataset(self):

        train_x1_file_name = os.path.join(self.dataset_path, 'XRMB[JW11,numfr1=7,numfr2=7,fold0,training].dat')
        train_x2_file_name = os.path.join(self.dataset_path, 'MFCC[JW11,numfr1=7,numfr2=7,fold0,training].dat')

        self.trainset = (self.ReadBin(train_x1_file_name, 112).T, self.ReadBin(train_x2_file_name, 273).T)

        test_x1_file_name = os.path.join(self.dataset_path, 'XRMB[JW11,numfr1=7,numfr2=7,fold0,testing].dat')
        test_x2_file_name = os.path.join(self.dataset_path, 'MFCC[JW11,numfr1=7,numfr2=7,fold0,testing].dat')

        self.testset = (self.ReadBin(test_x1_file_name, 112).T, self.ReadBin(test_x2_file_name, 273).T)

        tuning_x1_file_name = os.path.join(self.dataset_path, 'XRMB[JW11,numfr1=7,numfr2=7,fold0,tuning].dat')
        tuning_x2_file_name = os.path.join(self.dataset_path, 'MFCC[JW11,numfr1=7,numfr2=7,fold0,tuning].dat')

        self.tuning = (self.ReadBin(tuning_x1_file_name, 112).T, self.ReadBin(tuning_x2_file_name, 273).T)

    def ReadBin(self, file_name, row_num, max_col_num=-1):

        f = open(file_name, 'rb')

        f.seek(0, os.SEEK_END)
        endPos = f.tell()
        f.seek(0, os.SEEK_SET)

        col_num = int(endPos / numpy.finfo(numpy.float64).nexp / row_num)

        if max_col_num > -1:
            col_num = min(col_num, max_col_num)

        set = numpy.ndarray([row_num, col_num], dtype=config.floatX)

        for i in xrange(col_num):
            for j in xrange(row_num):
                set[j, i] = struct.unpack('d', f.read(8))[0]

        f.close()

        return set


class XRBMSampledDataSet(DatasetBase):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        self._samples_train = int(data_set_parameters['samples_train'])
        self._samples_test = int(data_set_parameters['samples_test'])
        super(XRBMSampledDataSet, self).__init__(data_set_parameters)

    def build_dataset(self):

        train_x1 = numpy.ndarray((0, 112))
        train_x2 = numpy.ndarray((0, 273))
        for i in range(5):
            path = os.path.join(self.dataset_path, 'JW11_fold{0}'.format(i))

            train_x1_file_name = os.path.join(path, 'XRMB[JW11,numfr1=7,numfr2=7,fold{0},training].dat'.format(i))
            train_x2_file_name = os.path.join(path, 'MFCC[JW11,numfr1=7,numfr2=7,fold{0},training].dat'.format(i))

            train_x1 = numpy.vstack((train_x1, self.ReadBin(train_x1_file_name, 112).T))
            train_x2 = numpy.vstack((train_x2, self.ReadBin(train_x2_file_name, 273).T))

        indices = numpy.random.randint(0, train_x1.shape[0], self._samples_train)
        self.trainset = (train_x1[indices].astype(theano.config.floatX, copy=False),
                         train_x2[indices].astype(theano.config.floatX, copy=False))

        test_x1 = numpy.ndarray((0, 112))
        test_x2 = numpy.ndarray((0, 273))
        for i in range(5):
            path = os.path.join(self.dataset_path, 'JW11_fold{0}'.format(i))

            test_x1_file_name = os.path.join(path, 'XRMB[JW11,numfr1=7,numfr2=7,fold{0},testing].dat'.format(i))
            test_x2_file_name = os.path.join(path, 'MFCC[JW11,numfr1=7,numfr2=7,fold{0},testing].dat'.format(i))

            test_x1 = numpy.vstack((train_x1, self.ReadBin(test_x1_file_name, 112).T))
            test_x2 = numpy.vstack((train_x2, self.ReadBin(test_x2_file_name, 273).T))

        indices = numpy.random.randint(0, test_x1.shape[0], self._samples_test)
        self.testset = (test_x1[indices].astype(theano.config.floatX, copy=False),
                        test_x2[indices].astype(theano.config.floatX, copy=False))

        tuning_x1 = numpy.ndarray((0, 112))
        tuning_x2 = numpy.ndarray((0, 273))
        for i in range(5):
            path = os.path.join(self.dataset_path, 'JW11_fold{0}'.format(i))

            tuning_x1_file_name = os.path.join(path, 'XRMB[JW11,numfr1=7,numfr2=7,fold{0},tuning].dat'.format(i))
            tuning_x2_file_name = os.path.join(path, 'MFCC[JW11,numfr1=7,numfr2=7,fold{0},tuning].dat'.format(i))

            tuning_x1 = numpy.vstack((train_x1, self.ReadBin(tuning_x1_file_name, 112).T))
            tuning_x2 = numpy.vstack((train_x2, self.ReadBin(tuning_x2_file_name, 273).T))

        indices = numpy.random.randint(0, tuning_x1.shape[0], self._samples_test)
        self.tuning = (tuning_x1[indices].astype(theano.config.floatX, copy=False),
                       tuning_x2[indices].astype(theano.config.floatX, copy=False))

    def ReadBin(self, file_name, row_num, max_col_num=-1):

        f = open(file_name, 'rb')

        f.seek(0, os.SEEK_END)
        endPos = f.tell()
        f.seek(0, os.SEEK_SET)

        col_num = int(endPos / numpy.finfo(numpy.float64).nexp / row_num)

        if max_col_num > -1:
            col_num = min(col_num, max_col_num)

        set = numpy.ndarray([row_num, col_num], dtype=config.floatX)

        for i in xrange(col_num):
            for j in xrange(row_num):
                set[j, i] = struct.unpack('d', f.read(8))[0]

        f.close()

        return set


class XRBMDebug(XRBMDataSet):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(XRBMDebug, self).__init__(data_set_parameters)

    def build_dataset(self):
        train_x1_file_name = self.dataset_path + '/XRMB[JW11,numfr1=7,numfr2=7,fold0,training].dat'
        train_x2_file_name = self.dataset_path + '/MFCC[JW11,numfr1=7,numfr2=7,fold0,training].dat'

        self.trainset = (self.ReadBin(train_x1_file_name, 112, 5000), self.ReadBin(train_x2_file_name, 273, 5000))

        test_x1_file_name = self.dataset_path + '/XRMB[JW11,numfr1=7,numfr2=7,fold0,testing].dat'
        test_x2_file_name = self.dataset_path + '/MFCC[JW11,numfr1=7,numfr2=7,fold0,testing].dat'

        self.testset = (self.ReadBin(test_x1_file_name, 112, 700), self.ReadBin(test_x2_file_name, 273, 700))

        tuning_x1_file_name = self.dataset_path + '/XRMB[JW11,numfr1=7,numfr2=7,fold0,tuning].dat'
        tuning_x2_file_name = self.dataset_path + '/MFCC[JW11,numfr1=7,numfr2=7,fold0,tuning].dat'

        self.tuning = (self.ReadBin(tuning_x1_file_name, 112, 500), self.ReadBin(tuning_x2_file_name, 273, 500))


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


'''
class XRBMDataSetRCCA(DatasetBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(XRBMDataSetRCCA, self).__init__(data_set_parameters)

    def build_dataset(self):

        os.system('cat %s/xrmb* > xrmb.data' % self.dataset_path)
        xrmb = robjects.r('load')(self.dataset_path + '/xrmb.data')

        self.trainset = numpy.array(robjects.r['x_tr'])[:, 0:30000].astype(theano.config.floatX, copy=False), \
                        numpy.array(robjects.r['y_tr'])[:, 0:30000].astype(theano.config.floatX, copy=False)

        self.tuning = numpy.array(robjects.r['x_tr'])[:, 30000:40000].astype(theano.config.floatX, copy=False), \
                      numpy.array(robjects.r['y_tr'])[:, 30000:40000].astype(theano.config.floatX, copy=False)

        self.testset = numpy.array(robjects.r['x_te']).astype(theano.config.floatX, copy=False), \
                       numpy.array(robjects.r['y_te']).astype(theano.config.floatX, copy=False)
'''
