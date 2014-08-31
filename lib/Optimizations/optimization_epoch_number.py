from numpy import arange

from optimization_base import OptimizationBase

from MISC.container import ContainerRegisterMetaClass
from MISC.logger import OutputLog

class OptimizationEpochNumber(OptimizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set, optimization_parameters, hyper_parameters,  regularization_methods):
        super(OptimizationEpochNumber, self).__init__(data_set, optimization_parameters, hyper_parameters,  regularization_methods)

        self.start_value = int(optimization_parameters['start_value'])
        self.end_value = int(optimization_parameters['end_value'])
        self.step = int(optimization_parameters['step'])

    def perform_optimization(self, training_strategy):

        OutputLog().write('----------------------------------------------------------')
        OutputLog().write('epochs correlations cca_correlations time')


        hyper_parameters = self.hyper_parameters.copy()
        best_correlation = 0

        #Weight decay optimization
        for i in arange(self.start_value, self.end_value, self.step):

            hyper_parameters.epochs = int(i)

            correlation, execution_time = self.train(training_strategy=training_strategy, hyper_parameters=hyper_parameters)

            if correlation > best_correlation:
                best_correlation = correlation
                self.hyper_parameters.epochs = int(i)

            OutputLog().write('%f, %s, %f\n' % (i,
                                                correlation,
                                                execution_time))

        OutputLog().write('----------------------------------------------------------')

        return True