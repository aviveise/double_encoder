from numpy.random import RandomState

from optimization_base import OptimizationBase

from MISC.container import ContainerRegisterMetaClass
from MISC.utils import print_list
from MISC.logger import OutputLog

class OptimizationLayerStructure(OptimizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set, optimization_parameters, hyper_parameters,  regularization_methods, top=50):
        super(OptimizationLayerStructure, self).__init__(data_set, optimization_parameters, hyper_parameters,  regularization_methods, top)

        self.symmetric_layers = bool(int(optimization_parameters['symmetric']))
        self.start_value = int(optimization_parameters['start_value'])
        self.end_value = int(optimization_parameters['end_value'])
        self.rounds_number = int(optimization_parameters['rounds_number'])
        self.layer_number = int(optimization_parameters['hidden_layer_number'])

    def perform_optimization(self, training_strategy):

        OutputLog().write('----------------------------------------------------------')
        OutputLog().write('layer_sizes correlations cca_correlations time')

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

            correlation, execution_time = self.train(training_strategy=training_strategy, hyper_parameters=hyper_parameters)

            if correlation > best_correlation:

                best_correlation = correlation
                self.hyper_parameters.layer_sizes = hyper_parameters.layer_sizes

            OutputLog().write('%s, %f, %f\n' % (print_list(hyper_parameters.layer_sizes),
                                                correlation,
                                                execution_time))


        OutputLog().write('----------------------------------------------------------')

        return True