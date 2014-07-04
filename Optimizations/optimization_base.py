import abc
import traceback
from time import clock

from numpy.random import RandomState
from theano.tensor.nnet import sigmoid

from MISC.utils import print_list
from training_strategy.iterative_training_strategy import IterativeTrainingStrategy
from correlation_test import CorrelationTest
from testers.double_encoder_tester import DoubleEncoderTester


__author__ = 'aviv eisenschtat'


class OptimizationBase(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, data_set, optimization_parameters, hyper_parameters,  regularization_methods, output_file):

        if hyper_parameters is None:
            raise ValueError('hyper parameters cannot be none')

        if data_set is None:
            raise ValueError('dataset cannot be none')

        self.training_set = data_set.trainset
        self.test_set = data_set.tuning
        self.hyper_parameters = hyper_parameters
        self.regularization_methods = regularization_methods
        self.output_file = output_file
        self.random_range = RandomState()

        # self.output_file = output_file

    def train(self, hyper_parameters=None, print_results=True, regularization_methods=None):

        start = clock()

        if hyper_parameters is None:
            hyper_parameters = self.hyper_parameters

        if regularization_methods is None:
            regularization_methods = self.regularization_methods

        try:
            training_strategy = IterativeTrainingStrategy(self.training_set[0].T,
                                                          self.training_set[1].T,
                                                          hyper_parameters, regularization_methods.values(), sigmoid)

            double_encoder = training_strategy.start_training()

            correlation_test = CorrelationTest(self.test_set[0].T, self.test_set[0].T)
            correlation = correlation_test.test(DoubleEncoderTester(double_encoder, 1))

        except:

            # self.output_file.write(traceback.format_exc())
            # self.output_file.flush()
            raise


        execution_time = clock() - start

        if print_results:
            self.print_results(hyper_parameters.layer_sizes, correlation, execution_time)

        return correlation, execution_time

    def print_results(self, layer_size, correlations, execution_time):

        # self.output_file.write('%s, %f, %s, %f\n' % (print_list(layer_size),
        #                                              correlations,
        #                                              print_list(correlations[1]),
        #                                              execution_time))

        # self.output_file.flush()


    @abc.abstractmethod
    def perform_optimization(self):
        """main optimization method"""
        return
