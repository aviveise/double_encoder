import os
import sys

from numpy import arange
from theano import config
from optimization_base import OptimizationBase
from MISC.container import ContainerRegisterMetaClass

class OptimizationRegularizationWeight(OptimizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set, optimization_parameters, hyper_parameters,  regularization_methods):
        super(OptimizationRegularizationWeight, self).__init__(data_set, optimization_parameters, hyper_parameters,  regularization_methods)

        self.start_value = optimization_parameters['start_value']
        self.end_value = optimization_parameters['end_value']
        self.step = optimization_parameters['step']

    def perform_optimization(self):

        self.output_file.write('Learning regularization\n')
        self.output_file.write('weight layer_sizes correlations cca_correlations time\n')
        self.output_file.flush()

        regularization_methods = self.regularization_methods.copy()
        best_correlation = 0

        #Weight decay optimization
        for i in arange(self.start_value,
                        self.end_value,
                        self.step,
                        dtype=config.floatX):

            regularization_methods['weight_decay_regularization'].regularization_parameters['weight'] = i

            self.output_file.write('%f, ' % i)
            correlations = self.train(regularization_methods=regularization_methods)

            if correlations[0] > best_correlation:
                best_correlation = correlations[0]
                self.regularization_methods['weight_decay_regularization'].regularization_parameters['weight'] = i

        return True