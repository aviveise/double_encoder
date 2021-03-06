from numpy import arange
from theano import config
from optimization_base import OptimizationBase

from MISC.container import ContainerRegisterMetaClass
from MISC.logger import OutputLog

class OptimizationBatchSize(OptimizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set, optimization_parameters, hyper_parameters, regularization_methods, top=50):
        super(OptimizationBatchSize, self).__init__(data_set, optimization_parameters, hyper_parameters, regularization_methods, top)

        self.start_value = float(optimization_parameters['start_value'])
        self.end_value = float(optimization_parameters['end_value'])
        self.step = float(optimization_parameters['step'])

    def perform_optimization(self, training_strategy):

        OutputLog().write('----------------------------------------------------------')
        OutputLog().write('batch_size correlations cca_correlations time')

        hyper_parameters = self.hyper_parameters.copy()

        best_correlation = 0

        #Weight decay optimization
        for i in arange(self.start_value,
                        self.end_value,
                        self.step,
                        dtype=config.floatX):

            batch_size = int(abs(self.training_set[0].shape[1] * i))

            hyper_parameters.batch_size = batch_size

            correlation, execution_time = self.train(training_strategy=training_strategy, hyper_parameters=hyper_parameters)

            if correlation > best_correlation:
                best_correlation = correlation
                self.hyper_parameters.batch_size = batch_size

            OutputLog().write('%f, %s, %f\n' % (batch_size,
                                                correlation,
                                                execution_time))

        OutputLog().write('----------------------------------------------------------')

        return True