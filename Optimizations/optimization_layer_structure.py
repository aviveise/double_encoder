import os
import sys

from numpy import arange
from numpy.random import RandomState

from optimization_base import OptimizationBase
from MISC.container import ContainerRegisterMetaClass

class OptimizationLayerStructure(OptimizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set, optimization_parameters, hyper_parameters,  regularization_methods, output_file):
        super(OptimizationLayerStructure, self).__init__(data_set, optimization_parameters, hyper_parameters,  regularization_methods, output_file)

        self.symmetric_layers = optimization_parameters['symmetric']
        self.start_value = optimization_parameters['start_value']
        self.end_value = optimization_parameters['end_value']
        self.rounds_number = optimization_parameters['rounds_number']
        self.layer_number = optimization_parameters['hidden_layer_number']

    def perform_optimization(self):

        hyper_parameters = self.hyper_parameters.copy()
        random_rng = RandomState()
        best_correlation = 0

        for iteration in xrange(int(self.rounds_number)):

            hyper_parameters.layer_sizes = random_rng.uniform(self.start_value,
                                                              self.end_value,
                                                              self.layer_number)

            hyper_parameters.layer_sizes = [int(round(layer_size)) for layer_size in hyper_parameters.layer_sizes]

            if self.symmetric_layers:
                for i in xrange(int(round(len(hyper_parameters.layer_sizes) / 2))):
                    hyper_parameters.layer_sizes[len(hyper_parameters.layer_sizes) - (i + 1)] = \
                        hyper_parameters.layer_sizes[i]

            correlations = self.train(hyper_parameters=hyper_parameters)

            if correlations[0] > best_correlation:

                best_correlation = correlations[0]
                self.hyper_parameters.layer_sizes = hyper_parameters.layer_sizes

        return True