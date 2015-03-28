import os
import sys
import ConfigParser
import scipy
import datetime

from numpy.random import RandomState

from time import clock

from configuration import Configuration

from Testers.trace_correlation_tester import TraceCorrelationTester

from Transformers.double_encoder_transformer import DoubleEncoderTransformer
from Transformers.gradient_transformer import GradientTransformer

from stacked_double_encoder import StackedDoubleEncoder

from MISC.container import Container
from MISC.utils import ConfigSectionMap
from MISC.logger import OutputLog

from Testers.trace_correlation_tester import TraceCorrelationTester

from Transformers.double_encoder_transformer import DoubleEncoderTransformer

import DataSetReaders
import Regularizations
import Optimizations

__author__ = 'aviv'


def run():

    data_set_config = sys.argv[1]
    run_time_config = sys.argv[2]
    double_encoder = sys.argv[3]
    top = int(sys.argv[4])
    outputs = sys.argv[5]

    if outputs == 'gradients':
        OutputLog().write('\nGradients is True\n')
        output_gradients = True
    elif outputs == 'activations':
        OutputLog().write('\nActivations is True\n')
        output_activations = True

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    #construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)

    #parse runtime configuration
    configuration = Configuration(run_time_config)
    configuration.hyper_parameters.batch_size = int(configuration.hyper_parameters.batch_size * data_set.trainset[0].shape[1])

    training_set_x = data_set.trainset[0].T
    training_set_y = data_set.trainset[1].T

    symmetric_double_encoder = StackedDoubleEncoder(hidden_layers=[],
                                                    numpy_range=RandomState(),
                                                    input_size=training_set_x.shape[1],
                                                    output_size=training_set_y.shape[1],
                                                    activation_method=None)

    symmetric_double_encoder.import_encoder(double_encoder, configuration.hyper_parameters)

    train_trace_correlation, x_train, y_train, train_best_layer = TraceCorrelationTester(data_set.trainset[0].T,
                                                                       data_set.trainset[1].T, top).test(DoubleEncoderTransformer(symmetric_double_encoder, 0),
                                                                                                         configuration.hyper_parameters)


    trace_correlation, x_test, y_test, test_best_layer = TraceCorrelationTester(data_set.testset[0].T,
                                                               data_set.testset[1].T, top).test(DoubleEncoderTransformer(symmetric_double_encoder, 0),
                                                                                                configuration.hyper_parameters)

    OutputLog().write('\nResults:\n')

    OutputLog().write('trace: correlation execution_time\n')

    OutputLog().write('%f\n' % float(trace_correlation))


    dir_name, filename = os.path.split(os.path.abspath(__file__))
    filename = outputs + '_' + data_parameters['name'] +'_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    OutputLog().write('output dir:' + dir_name)
    OutputLog().write('exporting double encoder:\n')

    export_test = {
        'best_layer': test_best_layer
    }

    export_train = {
        'best_layer': train_best_layer
    }

    if output_activations:

        OutputLog().write('\nOutput activations\n')

        for index in range(len(x_train)):

            set_name_x = 'hidden_train_x_%i' % index
            set_name_y = 'hidden_train_y_%i' % index
            export_train[set_name_x] = x_train[index]
            export_train[set_name_y] = y_train[index]

        for index in range(len(x_test)):

            set_name_x = 'hidden_test_x_%i' % index
            set_name_y = 'hidden_test_y_%i' % index
            export_test[set_name_x] = x_test[index]
            export_test[set_name_y] = y_test[index]

        scipy.io.savemat(os.path.join(dir_name, "train_" + filename + '.mat'), export_train)
        scipy.io.savemat(os.path.join(dir_name, "test_" + filename + '.mat'), export_test)


    if output_gradients:

        OutputLog().write('\nOutput gradients\n')

        transformer = GradientTransformer(symmetric_double_encoder, symmetric_double_encoder.getParams(), configuration.hyper_parameters)

        train_gradients = transformer.compute_outputs(data_set.trainset[0].T, data_set.trainset[1].T, 1)
        test_gradients = transformer.compute_outputs(data_set.testset[0].T, data_set.testset[1].T, 1)

        scipy.io.savemat(os.path.join(dir_name, "train_" + filename + '.mat'), train_gradients)
        scipy.io.savemat(os.path.join(dir_name, "test_" + filename + '.mat'), test_gradients)



if __name__ == '__main__':
    run()