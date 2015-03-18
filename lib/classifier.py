__author__ = 'aviv'
import os
import sys
import ConfigParser
import scipy.io
import traceback
import datetime
import numpy

from time import clock

from sklearn.svm import SVC

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

class Classifier(object):

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
                                                                       data_set.testset[1].T, top).test(DoubleEncoderTransformer(stacked_double_encoder, 0))


            transformer = GradientTransformer(stacked_double_encoder, stacked_double_encoder.getParams(), configuration.hyper_parameters)

            train_gradients = transformer.compute_outputs(data_set.trainset[0].T, data_set.trainset[1].T, 1)
            test_gradients = transformer.compute_outputs(data_set.testset[0].T, data_set.testset[1].T, 1)

            svm_classifier = SVC(kernel='linear')

            train_labels = numpy.ones((train_gradients.shape[0]))

            svm_classifier.fit(train_gradients, train_labels)

            test_labels = svm_classifier.predict(test_gradients)

            error = 1 - float(numpy.count_nonzero(test_labels)) / test_labels.shape[0]

            OutputLog().write('\nerror: %f\n' % error)


        except:
            print 'Exception: \n'
            print traceback.format_exc()
            raise



        return stacked_double_encoder


