from numpy.random import RandomState

from optimization_base import OptimizationBase

from MISC.container import ContainerRegisterMetaClass
from MISC.utils import print_list
from MISC.logger import OutputLog

class OptimizationLayerStructureSearch(OptimizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set, optimization_parameters, hyper_parameters,  regularization_methods, top=50):
        super(OptimizationLayerStructureSearch, self).__init__(data_set, optimization_parameters, hyper_parameters,  regularization_methods, top)

        self.symmetric_layers = bool(int(optimization_parameters['symmetric']))
        self.start_value = int(optimization_parameters['start_value'])
        self.end_value = int(optimization_parameters['end_value'])
        self.rounds_number = int(optimization_parameters['rounds_number'])
        self.layer_number_end = int(optimization_parameters['hidden_layer_number_end'])
        self.layer_number_start = int(optimization_parameters['hidden_layer_number_start'])

    def perform_optimization(self, training_strategy):

        OutputLog().write('----------------------------------------------------------')
        OutputLog().write('layer_sizes correlations cca_correlations time\n')

        hyper_parameters = self.hyper_parameters.copy()
        random_rng = RandomState()

        improvement_rounds = 0
        best_correlation = 0
        previous_correlation = 0
        for layer_num in xrange(int(self.layer_number_start),int(self.layer_number_end)):

            hyper_parameters.layer_sizes = random_rng.uniform(self.start_value,
                                                          self.end_value,
                                                          layer_size)

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
                                                                                                     layer_num)
                if correlation > best_correlation:
                    best_correlation = correlation
                    self.hyper_parameters.layer_sizes = hyper_parameters.layer_sizes

                OutputLog().write('%s, %f, %f\n' % (print_list(hyper_parameters.layer_sizes),
                                                    correlation,
                                                    execution_time))

        OutputLog().write('----------------------------------------------------------')

        return True