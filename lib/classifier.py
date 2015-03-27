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

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

class Classifier(object):



    @staticmethod
    def merge_gradients(gradients, layer):

        merged_gradients = None
        for sample in range(len(gradients)):

            sample_gradients = gradients[str(sample)]

            if layer == -1:

                descriptor = None
                for param in sample_gradients.keys():

                    print param

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

        train_gradients = loadmat(train_gradient_path)
        test_gradients = loadmat(test_gradient_path)

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


