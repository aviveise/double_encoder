    __author__ = 'aviv'

import sys
import os
import datetime

from theano.tensor.nnet import sigmoid

from training_strategy.iterative_training_strategy import IterativeTrainingStrategy
from MISC.container import Container
from configuration import Configuration

if __name__ == '__main__':

    data_set_config = sys.argv[1]
    run_time_config = sys.argv[2]

    training_strategy = IterativeTrainingStrategy()
    regularization_methods = {}

    output_file_name = 'double_encoder_' + str(datetime.datetime.now()) + '.txt'
    output_file = open(output_file_name, 'w+')

    #construct data set
    data_set = Container().create(data_set_config)

    #parse runtime configuration
    configuration = Configuration(run_time_config)

    #building regularization methods
    for regularization_parameters in configuration.regularizations_parameters:

        regularization_methods[regularization_parameters['type']] = Container().create(regularization_parameters['type'], regularization_parameters)

    #performing optimizations for various parameters
    for optimization_parameters in configuration.optimization_parameters:

        args = (optimization_parameters, configuration.hyper_parameters, regularization_methods, output_file)
        optimization = Container().create(optimization_parameters['type'], *args)
        optimization.perform_optimization(data_set, training_strategy,  configuration.hyper_parameters)

    #training the system with the optimized parameters
    stacked_double_encoder = training_strategy.train(training_set_x=data_set.training_set[0].T,
                                                     training_set_y=data_set.training_set[1].T,
                                                     hyper_parameters=configuration.hyper_parameters,
                                                     regularization_methods=regularization_methods.values(),
                                                     activation_method=sigmoid,
                                                     output_file=output_file)

    #testing the trained double encoder

    # output_file_name = 'double_encoder_' + str(datetime.datetime.now()) + '.txt'
    # output_file = open(output_file_name, 'w+')
    #
    # output_file.write('Starting Double Encoder\n')
    #
    # if not len(optimization_parameters) == 0:
    #
    #     output_file.write('Dataset = %s\n' % optimization_parameters[0].data_type)
    #
    #
    # for parameter in regular_parameters:
    #     run(parameter, output_file)
    #
    # output_file.close()
