import os
import sys

import cPickle
import gzip
import theano

from DataSetReaders.dataset_factory import dataset_meta
from DataSetReaders.dataset_base import DatasetBase

class MNISTDataSet(DatasetBase):

    __metaclass__ = dataset_meta

    def __init__(self, dataset_path, center=False, normalize=False, whiten=False):
        super(MNISTDataSet, self).__init__(dataset_path, 'MNIST', center, normalize, whiten)

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



