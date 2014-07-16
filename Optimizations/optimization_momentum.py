import os
import sys

from numpy import arange
from theano import config
from optimization_base import OptimizationBase

from MISC.container import ContainerRegisterMetaClass
from MISC.logger import OutputLog

class OptimizationMomentum(OptimizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set, optimization_parameters, hyper_parameters,  regularization_methods):
        super(OptimizationMomentum, self).__init__(optimization_parameters, hyper_parameters,  regularization_methods)

        self.start_value = float(optimization_parameters['start_value'])
        self.end_value = float(optimization_parameters['end_value'])
        self.step = float(optimization_parameters['step'])

    def perform_optimization(self, training_strategy):

        OutputLog().write('----------------------------------------------------------')
        OutputLog().write('batch_size layer_sizes correlations cca_correlations time\n')

        hyper_parameters = self.hyper_parameters.copy()
        best_correlation = 0

        #Weight decay optimization
        for i in arange(self.start_value,
                        self.end_value,
                        self.step,
                        dtype=config.floatX):

            hyper_parameters.momentum = i
            correlation, execution_time = self.train(training_strategy=training_strategy, hyper_parameters=hyper_parameters)

            if correlation > best_correlation:
                best_correlation = correlation
                self.hyper_parameters.batch_size = i

            OutputLog().write('%f, %s, %f\n' % (i,
                                                correlation,
                                                execution_time))

        OutputLog().write('----------------------------------------------------------')

        return True