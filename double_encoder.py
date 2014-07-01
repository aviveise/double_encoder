__author__ = 'aviv'

import sys
import os
import datetime

from double_encoder_parameters import DoubleEncoderParameters
from DataSetReaders.dataset_factory import DatasetFactory
from Optimizations.optimization_factory import OptimizationFactory
from theano.tensor.nnet import sigmoid
from iterative_training_strategy import IterativeTrainingStrategy
from Regularizations.weight_decay_regularization import WeightDecayRegularization

def run_optimizations(dataset, parameters, output_file):

    opt_factory = OptimizationFactory()

    args = (dataset, parameters, output_file)
    optimization = opt_factory.create(parameters.optimization_type, *args)

    output_file.write('\n--------------- Optimization Phase ---------------\n')
    output_file.write('Optimization type = %s\n' % parameters.optimization_type)

    return optimization.perform_optimization()

def run(parameters, output_file):

    data_factory = DatasetFactory()

    args = (parameters.dataset_path,
            parameters.center,
            parameters.normalize,
            parameters.whiten)
    dataset = data_factory.create(parameters.data_type, *args)


    output_file.write('Data type = %s\n' % parameters.data_type)
    output_file.write('Training set size = %d\n' % dataset.trainset[0].shape[1])
    output_file.write('Test set size = %d\n' % dataset.testset[0].shape[1])
    output_file.write('Tuning set size = %d\n' % dataset.tuning[0].shape[1])
    output_file.flush()

    regularization = WeightDecayRegularization(hyper_parameters.regularization_parameters['weight'])
    training_strategy = IterativeTrainingStrategy(dataset.trainset[0].T,
                                                  dataset.trainset[1].T,
                                                  hyper_parameters, regularization, sigmoid)
    training_strategy.start_training()

if __name__ == '__main__':

    path = sys.argv[1]

    config_files = []

    if os.path.isdir(path):
        config_files = [path + '/' + file for file in os.listdir(path) if file.endswith('.ini')]
    else:
        config_files.append(path)

    parameters = [DoubleEncoderParameters(config_file) for config_file in config_files]

    optimization_parameters = [parameter for parameter in parameters if parameter.optimization_mode]
    regular_parameters = [parameter for parameter in parameters if not parameter.optimization_mode]

    optimization_parameters = sorted(optimization_parameters, key=lambda parameter: parameter.optimization_priority)

    output_file_name = 'double_encoder_' + str(datetime.datetime.now()) + '.txt'
    output_file = open(output_file_name, 'w+')

    output_file.write('Starting Double Encoder\n')

    if not len(optimization_parameters) == 0:

        data_factory = DatasetFactory()

        args = (optimization_parameters[0].dataset_path,
                optimization_parameters[0].center,
                optimization_parameters[0].normalize,
                optimization_parameters[0].whiten)

        output_file.write('Dataset = %s\n' % optimization_parameters[0].data_type)
        dataset = data_factory.create(optimization_parameters[0].data_type, *args)

        hyper_parameters = optimization_parameters[0].base_hyper_parameters

        for parameter in optimization_parameters:

            parameter.base_hyper_parameters = hyper_parameters
            hyper_parameters = run_optimizations(dataset, parameter, output_file)


    for parameter in regular_parameters:
        run(parameter, output_file)

    output_file.close()
