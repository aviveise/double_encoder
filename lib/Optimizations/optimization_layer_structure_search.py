from numpy.random import RandomState

from lib.Optimizations.optimization_base import OptimizationBase

from lib.MISC.container import ContainerRegisterMetaClass
from lib.MISC.utils import print_list
from lib.MISC.logger import OutputLog

class OptimizationLayerStructureSearch(OptimizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set, optimization_parameters, hyper_parameters,  regularization_methods):
        super(OptimizationLayerStructureSearch, self).__init__(data_set, optimization_parameters, hyper_parameters,  regularization_methods)

        self.symmetric_layers = bool(int(optimization_parameters['symmetric']))
        self.start_value = int(optimization_parameters['start_value'])
        self.end_value = int(optimization_parameters['end_value'])
        self.rounds_number = int(optimization_parameters['rounds_number'])
        self.layer_number = int(optimization_parameters['hidden_layer_number'])

    def perform_optimization(self, training_strategy):

        OutputLog().write('----------------------------------------------------------')
        OutputLog().write('layer_sizes correlations cca_correlations time\n')

        hyper_parameters = self.hyper_parameters.copy()
        random_rng = RandomState()
        hyper_parameters.layer_sizes = random_rng.uniform(self.start_value,
                                                          self.end_value,
                                                          self.layer_number)

        improvement_rounds = 0
        best_correlation = 0
        previous_correlation = 0

        for iteration in xrange(int(self.rounds_number)):

            hyper_parameters.layer_sizes = [int(round(layer_size)) for layer_size in hyper_parameters.layer_sizes]

            correlation, execution_time = self.train(training_strategy=training_strategy, hyper_parameters=hyper_parameters)

            if correlation < previous_correlation:
                improvement_rounds = 0
                previous_correlation = correlation
            else:
                improvement_rounds += 1

            if improvement_rounds > 0:
                hyper_parameters.layer_sizes = [layer_size + 50 for layer_size in hyper_parameters.layer_sizes]

            else:
                hyper_parameters.layer_sizes = hyper_parameters.layer_sizes = random_rng.uniform(self.start_value,
                                                                                                 self.end_value,
                                                                                                 self.layer_number)
            if correlation > best_correlation:
                best_correlation = correlation
                self.hyper_parameters.layer_sizes = hyper_parameters.layer_sizes

            OutputLog().write('%s, %f, %f\n' % (print_list(hyper_parameters.layer_sizes),
                                                correlation,
                                                execution_time))

        OutputLog().write('----------------------------------------------------------')

        return True