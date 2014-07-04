import os
import sys

from numpy import arange
from numpy.random import RandomState

from optimization_base import OptimizationBase
from MISC.container import ContainerRegisterMetaClass

class OptimizationLayerStructure(OptimizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set, parameters, output_file, regularizations):
        super(OptimizationLayerStructure, self).__init__(data_set, parameters, output_file, regularizations)

        self.symmetric_layers = parameters.structure_optimization_symmetric
        self.layer_size_interval = parameters.optimization_interval
        self.layer_number = parameters.structure_optimization_layer_number

    def perform_optimization(self):

        self.output_file.write('Optimizing Layer Structure - Random\n')
        self.output_file.write('layer_sizes correlation cca_correlation time\n')

        self.output_file.flush()

        hyper_parameters = self.base_hyper_parameters.copy()
        random_rng = RandomState()
        best_correlation = 0

        for iteration in xrange(int(self.layer_size_interval.step)):

            hyper_parameters.layer_sizes = random_rng.uniform(self.layer_size_interval.start,
                                                              self.layer_size_interval.end,
                                                              self.layer_number)

            hyper_parameters.layer_sizes = [int(round(layer_size)) for layer_size in hyper_parameters.layer_sizes]

            if self.symmetric_layers:
                for i in xrange(int(round(len(hyper_parameters.layer_sizes) / 2))):
                    hyper_parameters.layer_sizes[len(hyper_parameters.layer_sizes) - (i + 1)] = \
                        hyper_parameters.layer_sizes[i]

            correlations = self.train(hyper_parameters=hyper_parameters)

            if correlations[0] > best_correlation:

                best_correlation = correlations[0]
                best_hyper_parameters = hyper_parameters.copy()

        return best_hyper_parameters