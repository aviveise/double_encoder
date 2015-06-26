import cPickle
import gzip
import theano

from MISC.container import ContainerRegisterMetaClass
from dataset_base import DatasetBase

import numpy

class RANDDataSet(DatasetBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(RANDDataSet, self).__init__(data_set_parameters)

    def build_dataset(self):

        self.numpy_range = numpy.random.RandomState()

        self.trainset = self.numpy_range.normal(loc=0.0, scale=1.0, size=[100, 10000]), \
                        self.numpy_range.normal(loc=0.0, scale=1.0, size=[100, 10000])

        self.testset = self.numpy_range.normal(loc=0.0, scale=1.0, size=[100, 1000]), \
                       self.numpy_range.normal(loc=0.0, scale=1.0, size=[100, 1000])

        self.tuning = self.numpy_range.normal(loc=0.0, scale=1.0, size=[100, 1000]), \
                      self.numpy_range.normal(loc=0.0, scale=1.0, size=[100, 1000])
