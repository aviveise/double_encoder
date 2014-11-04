__author__ = 'aviv'

import sys
import ConfigParser

import traceback

from time import clock

from configuration import Configuration

from Testers.trace_correlation_tester import TraceCorrelationTester
from Testers.cca_correlation_tester import CCACorraltionTester

from Transformers.double_encoder_transformer import DoubleEncoderTransformer

from MISC.container import Container
from MISC.utils import ConfigSectionMap
from MISC.logger import OutputLog

import DataSetReaders
import Regularizations

class DoubleEncoder(object):

    @staticmethod
    def run(training_strategy):

        data_set_config = sys.argv[1]
        run_time_config = sys.argv[2]
        top = int(sys.argv[3])

        regularization_methods = {}

        data_config = ConfigParser.ConfigParser()
        data_config.read(data_set_config)
        data_parameters = ConfigSectionMap("dataset_parameters", data_config)

        #construct data set
        data_set = Container().create(data_parameters['name'], data_parameters)

        #parse runtime configuration
        configuration = Configuration(run_time_config)

        configuration.hyper_parameters.batch_size = int(configuration.hyper_parameters.batch_size * data_set.trainset[0].shape[1])

        training_strategy.set_parameters(configuration.strategy_parameters)

        #building regularization methods
        for regularization_parameters in configuration.regularizations_parameters:

            regularization_methods[regularization_parameters['type']] = Container().create(regularization_parameters['type'], regularization_parameters)

        #performing optimizations for various parameters
        for optimization_parameters in configuration.optimizations_parameters:

            args = (data_set, optimization_parameters, configuration.hyper_parameters, regularization_methods, top)
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

            trace_correlation = TraceCorrelationTester(data_set.testset[0].T, data_set.testset[1].T, top).test(DoubleEncoderTransformer(stacked_double_encoder, 0))

            cca_correlation = CCACorraltionTester(data_set.testset[0].T,
                                                  data_set.testset[1].T,
                                                  data_set.trainset[0].T,
                                                  data_set.trainset[1].T, top).test(DoubleEncoderTransformer(stacked_double_encoder, 0))


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

        OutputLog().write('trace: correlation execution_time\n')

        OutputLog().write('%f, %f\n' % (float(trace_correlation),
                                          execution_time))

        OutputLog().write('cca: correlation execution_time\n')

        OutputLog().write('%f, %f\n' % (float(cca_correlation),
                                        execution_time))

        #x_test = data_set.testset[0][: ,0]
        #x_test.reshape([x_test.shape[0], 1])

        #y_test = data_set.testset[1][: ,0]
        #y_test.reshape([x_test.shape[0], 1])

        #print x_test.shape

        #x_tilde, y_tilde = DoubleEncoderTransformer(stacked_double_encoder,0).compute_reconstructions(x_test,
        #                                                                                              y_test)

        #image_x = Image.fromarray(x_tilde)
        #image_y = Image.fromarray(y_tilde)

        #image_x.save('x_tilde.png')
        #image_y.save('y_tilde.png')

        return stacked_double_encoder
