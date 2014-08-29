from numpy import arange
from theano import config
from lib.Optimizations.optimization_base import OptimizationBase

from lib.MISC.container import ContainerRegisterMetaClass
from lib.MISC.logger import OutputLog

class OptimizationRegularizationWeight(OptimizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set, optimization_parameters, hyper_parameters,  regularization_methods):
        super(OptimizationRegularizationWeight, self).__init__(data_set, optimization_parameters, hyper_parameters,  regularization_methods)

        self.start_value = float(optimization_parameters['start_value'])
        self.end_value = float(optimization_parameters['end_value'])
        self.step = float(optimization_parameters['step'])

    def perform_optimization(self, training_strategy):

        OutputLog().write('----------------------------------------------------------')
        OutputLog().write('weight correlations cca_correlations time\n')

        regularization_methods = self.regularization_methods.copy()
        best_correlation = 0

        #Weight decay optimization
        for i in arange(self.start_value,
                        self.end_value,
                        self.step,
                        dtype=config.floatX):

            regularization_methods['WeightDecayRegularization'].weight = i

            correlation, execution_time = self.train(training_strategy=training_strategy, regularization_methods=regularization_methods)

            if correlation > best_correlation:
                best_correlation = correlation
                self.regularization_methods['WeightDecayRegularization'].weight = i

            OutputLog().write('%f, %s, %f\n' % (i,
                                                correlation,
                                                execution_time))

        OutputLog().write('----------------------------------------------------------')

        return True