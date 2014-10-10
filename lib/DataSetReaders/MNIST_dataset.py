import cPickle
import gzip
import theano

from MISC.container import ContainerRegisterMetaClass
from dataset_base import DatasetBase

import numpy
import rpy2.robjects as robjects

class MNISTDataSet(DatasetBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(MNISTDataSet, self).__init__(data_set_parameters)

    def build_dataset(self):

        f = gzip.open(self.dataset_path, 'rb')
        train_set, test_set = cPickle.load(f)
        f.close()

        self.trainset = train_set[0].T.astype(theano.config.floatX, copy=False), \
                        train_set[1].T.astype(theano.config.floatX, copy=False)

        self.testset = test_set[0].T.astype(theano.config.floatX, copy=False), \
                       test_set[1].T.astype(theano.config.floatX, copy=False)

        x1_train_set, x1_tuning_set, test_samples = self.produce_optimization_sets(self.trainset[0])
        x2_train_set, x2_tuning_set, test_samples = self.produce_optimization_sets(self.trainset[1], test_samples)

        self.trainset = x1_train_set, x2_train_set
        self.tuning = x1_tuning_set, x2_tuning_set

class MNISTDataSetRCCA(DatasetBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(MNISTDataSet, self).__init__(data_set_parameters)

    def build_dataset(self):

        mnist = robjects.r('load')(self.dataset_path)

        self.trainset = numpy.array(robjects.r['x_tr']), numpy.array(robjects.r['y_tr'])

        self.testset = numpy.array(robjects.r['x_te']), numpy.array(robjects.r['y_te'])

        x1_train_set, x1_tuning_set, test_samples = self.produce_optimization_sets(self.trainset[0])
        x2_train_set, x2_tuning_set, test_samples = self.produce_optimization_sets(self.trainset[1], test_samples)

        self.trainset = x1_train_set, x2_train_set
        self.tuning = x1_tuning_set, x2_tuning_set