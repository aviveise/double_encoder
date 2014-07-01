import os
import sys

from numpy import arange
from theano import config
from optimization_base import OptimizationBase
from Optimizations.optimization_factory import OptimizationMetaClass

class OptimizationBatchSize(OptimizationBase):

    __metaclass__ = OptimizationMetaClass

    def __init__(self, data_set, parameters, output_file):
        super(OptimizationBatchSize, self).__init__(data_set, parameters, output_file)

        self.batch_size_interval = parameters.optimization_interval

    def perform_optimization(self):

        self.output_file.write('Learning Batch Sizes\n')
        self.output_file.write('batch_size layer_sizes correlations cca_correlations time\n')

        self.output_file.flush()

        hyper_parameters = self.base_hyper_parameters.copy()

        best_correlation = 0

        #Weight decay optimization
        for i in arange(self.batch_size_interval.start,
                        self.batch_size_interval.end,
                        self.batch_size_interval.step,
                        dtype = config.floatX):

            hyper_parameters.batch_size = int(i)

            self.output_file.write('%f, ' % i)
            correlations = self.train(hyper_parameters=hyper_parameters)

            if correlations[0] > best_correlation:

                best_correlation = correlations[0]
                best_hyper_parameters = hyper_parameters.copy()

        return best_hyper_parameters
