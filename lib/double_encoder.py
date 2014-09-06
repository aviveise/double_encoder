__author__ = 'aviv'

import sys
import ConfigParser

import DataSetReaders
import Regularizations
import Optimizations

import traceback
from time import clock

from theano.tensor.nnet import sigmoid

from Testers.double_encoder_tester import DoubleEncoderTester

from correlation_test import CorrelationTest

from configuration import Configuration

from MISC.container import Container
from MISC.utils import ConfigSectionMap
from MISC.logger import OutputLog

class DoubleEncoder(object):

    @staticmethod
    def run(training_strategy):

        data_set_config = sys.argv[1]
        run_time_config = sys.argv[2]

        regularization_methods = {}

        data_config = ConfigParser.ConfigParser()
        data_config.read(data_set_config)
        data_parameters = ConfigSectionMap("dataset_parameters", data_config)

        #construct data set
        data_set = Container().create(data_parameters['name'], data_parameters)

        #parse runtime configuration
        configuration = Configuration(run_time_config)

        configuration.hyper_parameters.batch_size = int(configuration.hyper_parameters.batch_size * data_set.trainset[0].shape[1])

        #building regularization methods
        for regularization_parameters in configuration.regularizations_parameters:

            regularization_methods[regularization_parameters['type']] = Container().create(regularization_parameters['type'], regularization_parameters)

        #performing optimizations for various parameters
        for optimization_parameters in configuration.optimizations_parameters:

            args = (data_set, optimization_parameters, configuration.hyper_parameters, regularization_methods)
            optimization = Container().create(optimization_parameters['type'], *args)
            optimization.perform_optimization(training_strategy)

        start = clock()

        try:

            #training the system with the optimized parameters
            stacked_double_encoder = training_strategy.train(training_set_x=data_set.trainset[0].T,
                                                             training_set_y=data_set.trainset[1].T,
                                                             hyper_parameters=configuration.hyper_parameters,
                                                             regularization_methods=regularization_methods.values(),
                                                             activation_method=None)

            correlation = CorrelationTest(data_set.testset[0].T, data_set.testset[1].T).test(DoubleEncoderTester(stacked_double_encoder, 1))

        except:
            print 'Exception: \n'
            print traceback.format_exc()
            raise

        execution_time = clock() - start

        OutputLog().write('\nTest results : \n')

        configuration.hyper_parameters.print_parameters(OutputLog())

        OutputLog().write('\nRegularizations:')

        for regularization in regularization_methods.values():
            regularization.print_regularization(OutputLog())

        OutputLog().write('\nResults:\n')

        OutputLog().write('correlation execution_time\n')

        OutputLog().write('%f%%, %f\n' % (float(correlation) * 100,
                                          execution_time))


        return stacked_double_encoder
