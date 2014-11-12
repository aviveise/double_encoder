import abc
import traceback
from time import clock

from numpy.random import RandomState
from theano.tensor.nnet import sigmoid

from MISC.logger import OutputLog
from TrainingStrategy.iterative_training_strategy import IterativeTrainingStrategy
from Testers.trace_correlation_tester import TraceCorrelationTester
from Transformers.double_encoder_transformer import DoubleEncoderTransformer


__author__ = 'aviv eisenschtat'


class OptimizationBase(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, data_set, optimization_parameters, hyper_parameters, regularization_methods, top=50):

        OutputLog().write('\nCreating optimization: ' + optimization_parameters['type'])

        if hyper_parameters is None:
            raise ValueError('hyper parameters cannot be none')

        if data_set is None:
            raise ValueError('dataset cannot be none')

        self.training_set = data_set.trainset
        self.test_set = data_set.tuning
        self.hyper_parameters = hyper_parameters
        self.regularization_methods = regularization_methods
        self.random_range = RandomState()
        self.top = top

    def train(self, training_strategy=IterativeTrainingStrategy, hyper_parameters=None, regularization_methods=None):

        start = clock()

        if hyper_parameters is None:
            hyper_parameters = self.hyper_parameters

        if regularization_methods is None:
            regularization_methods = self.regularization_methods

        try:
            double_encoder = training_strategy.train(self.training_set[0].T,
                                                     self.training_set[1].T,
                                                     hyper_parameters,
                                                     regularization_methods.values(),
                                                     sigmoid,
                                                     top=self.top)

            correlation = TraceCorrelationTester(self.test_set[0].T, self.test_set[1].T, self.top).\
                test(DoubleEncoderTransformer(double_encoder, 0))

        except:

            print traceback.format_exc()
            raise

        execution_time = clock() - start

        return correlation, execution_time

    @abc.abstractmethod
    def perform_optimization(self, training_strategy):
        """main optimization method"""
        return
