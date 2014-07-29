__author__ = 'aviv'

import sys
import os
import datetime
import ConfigParser

from theano.tensor.nnet import sigmoid
from training_strategy.iterative_training_strategy import IterativeTrainingStrategy

from Optimizations.optimization_base import OptimizationBase
from DataSetReaders.dataset_base import DatasetBase
from Regularizations.regularization_base import RegularizationBase

from configuration import Configuration

from MISC.container import Container
from MISC.utils import ConfigSectionMap

if __name__ == '__main__':

    data_set_config = sys.argv[1]
    run_time_config = sys.argv[2]

    training_strategy = IterativeTrainingStrategy()
    regularization_methods = {}

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    #construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)

    #parse runtime configuration
    configuration = Configuration(run_time_config)

    configuration.hyper_parameters.batch_size = configuration.hyper_parameters.batch_size * data_set.training_set[0].shape[0]

    #building regularization methods
    for regularization_parameters in configuration.regularizations_parameters:

        regularization_methods[regularization_parameters['type']] = Container().create(regularization_parameters['type'], regularization_parameters)

    #performing optimizations for various parameters
    for optimization_parameters in configuration.optimizations_parameters:

        args = (data_set, optimization_parameters, configuration.hyper_parameters, regularization_methods)
        optimization = Container().create(optimization_parameters['type'], *args)
        optimization.perform_optimization(training_strategy)

    #training the system with the optimized parameters
    stacked_double_encoder = training_strategy.train(training_set_x=data_set.training_set[0].T,
                                                     training_set_y=data_set.training_set[1].T,
                                                     hyper_parameters=configuration.hyper_parameters,
                                                     regularization_methods=regularization_methods.values(),
                                                     activation_method=sigmoid)
