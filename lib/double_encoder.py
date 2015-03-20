__author__ = 'aviv'
import os
import sys
import ConfigParser
import scipy.io
import traceback
import datetime

from time import clock

from configuration import Configuration

from Testers.trace_correlation_tester import TraceCorrelationTester

from Transformers.double_encoder_transformer import DoubleEncoderTransformer

from MISC.container import Container
from MISC.utils import ConfigSectionMap
from MISC.logger import OutputLog

import DataSetReaders
import Regularizations
import Optimizations

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
                                                             activation_method=None,
                                                             top=top,
                                                             print_verbose=True,
                                                             validation_set_x=data_set.tuning[0],
                                                             validation_set_y=data_set.tuning[1])

            trace_correlation, x_best, y_best = TraceCorrelationTester(data_set.testset[0].T,
                                                                       data_set.testset[1].T, top).test(DoubleEncoderTransformer(stacked_double_encoder, 0),
                                                                                                        configuration.hyper_parameters)


            train_trace_correlation, x_train_best, y_train_best = TraceCorrelationTester(data_set.trainset[0].T,
                                                                                         data_set.trainset[1].T, top).test(DoubleEncoderTransformer(stacked_double_encoder, 0),
                                                                                                                           configuration.hyper_parameters)

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

        dirname, filename = os.path.split(os.path.abspath(__file__))
        filename = data_parameters['name'] + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.mat'

        export_test = {
            'image_decca_test': x_best,
            'sent_decca_test': y_best
        }

        export_train = {
            'image_decca_train': x_train_best,
            'sent_decca_train': y_train_best
        }

        try:
            scipy.io.savemat(os.path.join(dirname, "train_" + filename), export_train)
            scipy.io.savemat(os.path.join(dirname, "test_" + filename), export_test)
        except:
            'exporting to mat file failed'

        return stacked_double_encoder


