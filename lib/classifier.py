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

        output_gradients = None
        samples = gradients.keys()
        samples.sort()
        for sample in samples:

            sample_gradients = gradients[sample]

            try:

                if layer == -1:

                    descriptor = None
                    for param in sample_gradients.keys():

                        if param[0] == 'W':

                            if descriptor is None:
                                descriptor = sample_gradients[param].flatten()
                            else:
                                descriptor = numpy.concatenate((descriptor, sample_gradients[param].flatten()))

                else:
                    descriptor = numpy.concatenate((sample_gradients['Wx_layer' + str(layer).flatten()],
                                                    sample_gradients['Wy_layer' + str(layer).flatten()]))


                if output_gradients is None:
                    output_gradients = descriptor
                else:
                    output_gradients = numpy.concatenate((output_gradients, descriptor), axis=0)

            except:
                OutputLog().write('failed processing sample: ' + sample)

        return output_gradients


    @staticmethod
    def run():

        train_gradient_path = sys.argv[1]
        test_gradient_path = sys.argv[2]
        layer = int(sys.argv[3])

        train_gradients = loadmat(train_gradient_path)
        test_gradients = loadmat(test_gradient_path)

        train_gradients = Classifier.merge_gradients(train_gradients, layer)
        test_gradients = Classifier.merge_gradients(test_gradients, layer)

        OutputLog().write('Processed training set, sized: [%d, %d]' % (train_gradients.shape[0], train_gradients.shape[1]))
        OutputLog().write('Processed test set, sized: [%d, %d]' % (test_gradients.shape[0], test_gradients.shape[1]))

        svm_classifier = LinearSVC()

        train_labels = numpy.arange(10)
        for i in range(train_gradients.shape[0] / 10 - 1):
            train_labels = numpy.concatenate((train_labels, numpy.arange(10)))

        test_labels = numpy.arange(10)
        for i in range(test_gradients.shape[0] / 10):
            test_labels = numpy.concatenate((test_labels, numpy.arange(10)))

        svm_classifier.fit(train_gradients, train_labels)

        test_labels = svm_classifier.predict(test_gradients)

        error = 1 - float(numpy.count_nonzero(test_labels)) / test_labels.shape[0]

        OutputLog().write('\nerror: %f\n' % error)

        return stacked_double_encoder


