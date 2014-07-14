import os
import sys

from numpy import arange
from theano import config
from optimization_base import OptimizationBase

from MISC.container import ContainerRegisterMetaClass
from MISC.logger import OutputLog

class OptimizationRegularizationWeight(OptimizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set, optimization_parameters, hyper_parameters,  regularization_methods):
        super(OptimizationRegularizationWeight, self).__init__(data_set, optimization_parameters, hyper_parameters,  regularization_methods)

        self.start_value = optimization_parameters['start_value']
        self.end_value = optimization_parameters['end_value']
        self.step = optimization_parameters['step']

    def perform_optimization(self):

        OutputLog().write('----------------------------------------------------------')
        OutputLog().write('batch_size layer_sizes correlations cca_correlations time\n')

        regularization_methods = self.regularization_methods.copy()
        best_correlation = 0

        #Weight decay optimization
        for i in arange(self.start_value,
                        self.end_value,
                        self.step,
                        dtype=config.floatX):

            regularization_methods['weight_decay_regularization'].regularization_parameters['weight'] = i

            correlation, execution_time = self.train(regularization_methods=regularization_methods)

            if correlation > best_correlation:
                best_correlation = correlation
                self.regularization_methods['weight_decay_regularization'].regularization_parameters['weight'] = i

            OutputLog().write('%f, %s, %f\n' % (i,
                                                correlation,
                                                execution_time))

        OutputLog().write('----------------------------------------------------------')

        return True