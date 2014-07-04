import os
import sys

from numpy import arange

from optimization_base import OptimizationBase
from MISC.container import ContainerRegisterMetaClass

class OptimizationEpochNumber(OptimizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set, optimization_parameters, hyper_parameters,  regularization_methods, output_file):
        super(OptimizationEpochNumber, self).__init__(data_set, optimization_parameters, hyper_parameters,  regularization_methods, output_file)

        self.start_value = optimization_parameters['start_value']
        self.end_value = optimization_parameters['end_value']
        self.step = optimization_parameters['step']

    def perform_optimization(self):

        self.output_file.write('Learning epochs\n')
        self.output_file.write('epochs_number layer_sizes correlations cca_correlations time\n')

        self.output_file.flush()

        hyper_parameters = self.hyper_parameters.copy()
        best_correlation = 0

        #Weight decay optimization
        for i in arange(self.start_value, self.end_value, self.step):

            hyper_parameters.epochs = int(i)

            self.output_file.write('%f, ' % i)
            correlations = self.train(hyper_parameters=hyper_parameters)

            if correlations[0] > best_correlation:
                best_correlation = correlations[0]
                self.hyper_parameters.batch_size = int(i)

        return True