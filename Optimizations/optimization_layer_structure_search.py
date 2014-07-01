import os
import sys

from numpy import arange
from numpy.random import RandomState

from optimization_base import OptimizationBase
from Optimizations.optimization_factory import OptimizationMetaClass

class OptimizationLayerStructureSearch(OptimizationBase):

    __metaclass__ = OptimizationMetaClass

    def __init__(self, data_set, parameters, output_file):
        super(OptimizationLayerStructureSearch, self).__init__(data_set, parameters, output_file)

        self.symmetric_layers = parameters.structure_optimization_symmetric
        self.layer_size_interval = parameters.optimization_interval
        self.layer_number = parameters.structure_optimization_layer_number

    def perform_optimization(self):

        self.output_file.write('Optimizing Layer Structure - Search\n')
        self.output_file.write('layer_sizes correlation cca_correlation time\n')

        self.output_file.flush()

        hyper_parameters = self.base_hyper_parameters.copy()

        correlation = 0
        random_rng = RandomState()

        hyper_parameters.layer_sizes = hyper_parameters.layer_sizes = random_rng.uniform(self.layer_size_interval.start,
                                                                                         self.layer_size_interval.end,
                                                                                         self.layer_number)

        improvement_rounds = 0
        best_correlation = 0

        for iteration in xrange(int(self.layer_size_interval.step)):

            hyper_parameters.layer_sizes = [int(round(layer_size)) for layer_size in hyper_parameters.layer_sizes]

            correlations, execution_time = self.train(hyper_parameters=hyper_parameters, print_results=False)
            relative_correlation = correlations[0][0]

            self.print_results(hyper_parameters.layer_sizes, correlations, execution_time)

            if relative_correlation > correlation:
                improvement_rounds = 0
                correlation = relative_correlation

            else:
                improvement_rounds += 1

            if improvement_rounds < 10:
                hyper_parameters.layer_sizes = [layer_size + 50 for layer_size in hyper_parameters.layer_sizes]

            else:
                hyper_parameters.layer_sizes = hyper_parameters.layer_sizes = random_rng.uniform(self.layer_size_interval.start,
                                                                                                 self.layer_size_interval.end,
                                                                                                 self.layer_number)


            if correlations[0] > best_correlation:

                best_correlation = correlations[0]
                best_hyper_parameters = hyper_parameters.copy()

        return best_hyper_parameters