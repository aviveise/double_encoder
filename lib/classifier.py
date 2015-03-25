__author__ = 'aviv'
import os
import sys
import ConfigParser
import scipy.io
import traceback
import datetime
import numpy
import cPickle


from time import clock

from sklearn.svm import SVC, LinearSVC

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
import numpy

class Classifier(object):

    @staticmethod
    def merge_gradients(gradients, layer):

        merged_gradients = None
        for sample in rang(len(gradients.keys())):

            sample_gradients = gradients[str(sample)]

            if layer == -1:

                descriptor = None
                for param in sample_gradients.keys():

                    if param[0] == 'W':

                        if descriptor is None:
                            descriptor = sample_gradients[param]
                        else:
                            numpy.concatenate((descriptor, sample_gradients[param].flatten()))

            else:
                descriptor = numpy.concatenate((sample_gradients['Wx_layer' + str(layer).flatten()],
                                                sample_gradients['Wy_layer' + str(layer).flatten()]))

            if merged_gradients is None:
                merged_gradients = descriptor
            else:
                numpy.concatenate((merged_gradients, descriptor), axis=0)
        return merged_gradients

    @staticmethod
    def run():

        train_gradient_path = sys.argv[1]
        test_gradient_path = sys.argv[2]
        layer = int(sys.argv[3])

        train_gradients = scipy.io.loadmat(file(train_gradient_path, 'rb'))
        #test_gradients = scipy.io.loadmat(file(test_gradient_path, 'rb'))

        train_gradients = Classifier.merge_gradients(train_gradients, layer)
        test_gradients = Classifier.merge_gradients(test_gradients, layer)

        svm_classifier = LinearSVC()

        train_labels = []
        for i in range(train_gradients.shape[0] / 10):
            train_labels += range(10)

        test_labels = []
        for i in range(test_gradients.shape[0] / 10):
            test_labels += range(10)

        svm_classifier.fit(train_gradients, train_labels)

        test_labels = svm_classifier.predict(test_gradients)

        error = 1 - float(numpy.count_nonzero(test_labels)) / test_labels.shape[0]

        OutputLog().write('\nerror: %f\n' % error)

        return stacked_double_encoder


