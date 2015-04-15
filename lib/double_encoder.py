import numpy

__author__ = 'aviv'
import os
import sys
import ConfigParser
import scipy.io
import traceback
import datetime
import cPickle

from time import clock

from configuration import Configuration

from Testers.trace_correlation_tester import TraceCorrelationTester

from Transformers.double_encoder_transformer import DoubleEncoderTransformer
from Transformers.gradient_transformer import GradientTransformer

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
        encoder_type = sys.argv[4]

        dir_name, filename = os.path.split(os.path.abspath(__file__))

        regularization_methods = {}

        data_config = ConfigParser.ConfigParser()
        data_config.read(data_set_config)
        data_parameters = ConfigSectionMap("dataset_parameters", data_config)

        #construct data set
        data_set = Container().create(data_parameters['name'], data_parameters)

        #parse runtime configuration
        configuration = Configuration(run_time_config)

        OutputLog().set_path(configuration.output_parameters['path'])

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
                                                             print_verbose=False,
                                                             validation_set_x=data_set.tuning[0],
                                                             validation_set_y=data_set.tuning[1],
                                                             dir_name=dir_name,
                                                             encoder_type=encoder_type)

            OutputLog().write('test results:')
            trace_correlation, x_test, y_test, reconstructions_train, test_best_layer = TraceCorrelationTester(data_set.testset[0].T,
                                                                       data_set.testset[1].T, top).test(DoubleEncoderTransformer(stacked_double_encoder, 0),
                                                                                                        configuration.hyper_parameters)

            OutputLog().write('train results:')
            train_trace_correlation, x_train, y_train, reconstructions_test, train_best_layer = TraceCorrelationTester(data_set.trainset[0].T,
                                                                             data_set.trainset[1].T, top).test(DoubleEncoderTransformer(stacked_double_encoder, 0),
                                                                                                               configuration.hyper_parameters)

        except:
            OutputLog().write('\nExceptions:\n')
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


        filename = configuration.output_parameters['type'] + '_' + data_parameters['name'] +'_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        dir_name = configuration.output_parameters['path']

        OutputLog().write('output dir:' + dir_name)
        OutputLog().write('exporting double encoder:\n')

        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        stacked_double_encoder.export_encoder(dir_name)

        export_test = {
            'best_layer': test_best_layer
        }

        export_train = {
            'best_layer': train_best_layer
        }

        if configuration.output_parameters['type'] == 'activations':

            OutputLog().write('\nOutput activations\n')

            for index in range(len(x_train)):

                set_name_x = 'hidden_train_x_%i' % index
                set_name_y = 'hidden_train_y_%i' % index
                set_recon_x = 'recon_train_x_%i' % index
                set_recon_y = 'recon_train_y_%i' % index
                export_train[set_name_x] = x_train[index]
                export_train[set_name_y] = y_train[index]
                export_train[set_recon_x] = reconstructions_train[index * 2]
                export_train[set_recon_y] = reconstructions_train[index * 2 + 1]

            for index in range(len(x_test)):

                set_name_x = 'hidden_test_x_%i' % index
                set_name_y = 'hidden_test_y_%i' % index
                set_recon_x = 'recon_test_x_%i' % index
                set_recon_y = 'recon_test_y_%i' % index
                export_test[set_name_x] = x_test[index]
                export_test[set_name_y] = y_test[index]
                export_train[set_recon_x] = reconstructions_test[index * 2]
                export_train[set_recon_y] = reconstructions_test[index * 2 + 1]


            scipy.io.savemat(os.path.join(dir_name, "train_" + filename + '.mat'), export_train)
            scipy.io.savemat(os.path.join(dir_name, "test_" + filename + '.mat'), export_test)

        sample = configuration.output_parameters['sample']
        sample_number = configuration.output_parameters['sample_number']

        if configuration.output_parameters['type'] == 'gradients':

            OutputLog().write('\nOutput gradients\n')

            transformer = GradientTransformer(stacked_double_encoder, stacked_double_encoder.getParams(), configuration.hyper_parameters)

            train_dir_name = os.path.join(dir_name, 'train')

            if not os.path.isdir(train_dir_name):
                os.makedirs(train_dir_name)

            for ndx, gradient in enumerate(transformer.compute_outputs(data_set.trainset[0].T, data_set.trainset[1].T, 1)):
                if sample:
                    for param in gradient.keys():
                        if gradient[param].shape[0] > sample_number:
                            indices = numpy.random.uniform(0, gradient[param].shape[0], sample_number).astype(int)
                            gradient[param] = gradient[param][indices]
                scipy.io.savemat(os.path.join(train_dir_name, "train_gradients_sample_{0}.mat".format(ndx)), gradient)

            test_dir_name = os.path.join(dir_name, 'test')

            if not os.path.isdir(test_dir_name):
                os.makedirs(test_dir_name)

            for ndx, gradient in enumerate(transformer.compute_outputs(data_set.testset[0].T, data_set.testset[1].T, 1)):
                if sample:
                    for param in gradient.keys():
                        if gradient[param].shape[0] > sample_number:
                            indices = numpy.random.uniform(0, gradient[param].shape[0], sample_number).astype(int)
                            gradient[param] = gradient[param][indices]
                scipy.io.savemat(os.path.join(test_dir_name, "test_gradients_sample_{0}.mat".format(ndx)), gradient)

        return stacked_double_encoder


