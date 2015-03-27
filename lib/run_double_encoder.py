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

from stacked_double_encoder import StackedDoubleEncoder

from MISC.container import Container
from MISC.utils import ConfigSectionMap
from MISC.logger import OutputLog

from Testers.trace_correlation_tester import TraceCorrelationTester

from Transformers.double_encoder_transformer import DoubleEncoderTransformer

import DataSetReaders

__author__ = 'aviv'

def run():

    data_set_config = sys.argv[1]
    run_time_config = sys.argv[2]
    double_encoder = sys.argv[3]
    top = int(sys.argv[4])

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

    train_trace_correlation, x_train, y_train, test_best_layer = TraceCorrelationTester(data_set.trainset[0].T,
                                                                       data_set.trainset[1].T, top).test(DoubleEncoderTransformer(symmetric_double_encoder, 0),
                                                                                                         configuration.hyper_parameters)


    trace_correlation, x_test, t_test, test_best_layer = TraceCorrelationTester(data_set.testset[0].T,
                                                               data_set.testset[1].T, top).test(DoubleEncoderTransformer(symmetric_double_encoder, 0),
                                                                                                configuration.hyper_parameters)

    OutputLog().write('\nResults:\n')

    OutputLog().write('trace: correlation execution_time\n')

    OutputLog().write('%f\n' % float(trace_correlation))



if __name__ == '__main__':
    run()