import os
import sys

from numpy import arange
from theano import config
from optimization_base import OptimizationBase

from MISC.container import ContainerRegisterMetaClass
from MISC.logger import OutputLog

class OptimizationBatchSize(OptimizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set, optimization_parameters, hyper_parameters,  regularization_methods):
        super(OptimizationBatchSize, self).__init__(data_set, optimization_parameters, hyper_parameters,  regularization_methods)

        self.start_value = optimization_parameters['start_value']
        self.end_value = optimization_parameters['end_value']
        self.step = optimization_parameters['step']

    def perform_optimization(self):

        OutputLog().write('----------------------------------------------------------')
        OutputLog().write('batch_size layer_sizes correlations cca_correlations time')

        hyper_parameters = self.hyper_parameters.copy()

        best_correlation = 0

        #Weight decay optimization
        for i in arange(self.start_value,
                        self.end_value,
                        self.step,
                        dtype=config.floatX):

            hyper_parameters.batch_size = int(i)

            correlation, execution_time = self.train(hyper_parameters=hyper_parameters)

            if correlation > best_correlation:
                best_correlation = correlation
                self.hyper_parameters.batch_size = int(i)

            OutputLog().write('%f, %s, %f\n' % (i,
                                                correlation,
                                                execution_time))

        OutputLog().write('----------------------------------------------------------')

        return True