import gc
from sklearn.linear_model import SGDClassifier

__author__ = 'aviv'
import os
import sys
import ConfigParser
import scipy.io
import numpy

from numpy.random.mtrand import RandomState

from sklearn.svm import SVC, LinearSVC

from configuration import Configuration

from stacked_double_encoder import StackedDoubleEncoder
from Testers.trace_correlation_tester import TraceCorrelationTester

from Transformers.double_encoder_transformer import DoubleEncoderTransformer
from Transformers.gradient_transformer import GradientTransformer

from MISC.container import Container
from MISC.utils import ConfigSectionMap
from MISC.logger import OutputLog

import DataSetReaders
import Regularizations
import Optimizations

def lincompress(x):
    U, S, V = scipy.linalg.svd(numpy.dot(x.T, x))
    xc = numpy.dot(U, numpy.diag(numpy.sqrt(S))).T

    return xc

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


def test_transformer(transformer, data_set, configuration):

    clf = SGDClassifier()
    samples = []
    labels = range(10)
    for epoch in range(configuration.hyper_parameters.epochs):
        for index, sample in enumerate(transformer.compute_outputs(data_set.trainset[0].T, data_set.trainset[1].T, 1)):

            samples.extend(sample.reshape((1, sample.shape[0])))
            if index % 10 == 9:
                clf.partial_fit(samples, labels, labels)
                samples = []
                gc.collect()

    error = 0
    count = 0
    test_predictions = []
    for index, sample in enumerate(transformer_x.compute_outputs(data_set.testset[0].T, data_set.testset[1].T, 1)):
        prediction = clf.predict(sample)
        print prediction
        if not prediction == index % 10:
            error += 1

        count += 1
        test_predictions.extend(prediction)

    OutputLog().write('test predictions weight: {0}'.format(test_predictions))

    OutputLog().write('\nerror: %f%%\n' % error)

class Classifier(object):

    @staticmethod
    def merge_gradients(gradients):

        output_gradients = None
        samples = gradients.keys()
        samples.sort()
        for sample in samples:

            sample_gradients = gradients[sample]

            try:

                descriptor = None
                for param in sample_gradients.keys():
                        if descriptor is None:
                            descriptor = sample_gradients[param].flatten()
                        else:
                            descriptor = numpy.concatenate((descriptor, sample_gradients[param].flatten()))

                if output_gradients is None:
                    output_gradients = descriptor.reshape(1, descriptor.shape[0])
                else:
                    output_gradients = numpy.concatenate((output_gradients, descriptor.reshape(1, descriptor.shape[0])),
                                                         axis=0)

            except:
                OutputLog().write('failed processing sample: ' + sample)

            sample_gradients = 0
            gradients[sample] = 0
            gc.collect()

        return output_gradients


    @staticmethod
    def run():

        data_set_config = sys.argv[1]
        run_time_config = sys.argv[2]
        double_encoder = sys.argv[3]
        compress = int(sys.argv[4])
        layer = int(sys.argv[5])

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
                                                        batch_size=configuration.hyper_parameters.batch_size,
                                                        activation_method=None)

        symmetric_double_encoder.import_encoder(double_encoder, configuration.hyper_parameters)

        OutputLog().write('calculating gradients')

        params_x = [symmetric_double_encoder[layer].Wx]
        params_y = [symmetric_double_encoder[layer].Wy]

        print symmetric_double_encoder[layer].Wx.get_value(borrow=True).shape

        transformer_x = GradientTransformer(symmetric_double_encoder, params_x, configuration.hyper_parameters)
        transformer_y = GradientTransformer(symmetric_double_encoder, params_y, configuration.hyper_parameters)

        test_transformer(transformer_x, data_set, configuration)
        test_transformer(transformer_y, data_set, configuration)


        #train_gradients = transformer.compute_outputs(data_set.trainset[0].T, data_set.trainset[1].T, 1)
        #test_gradients = transformer.compute_outputs(data_set.testset[0].T, data_set.testset[1].T, 1)

        #OutputLog().write('Processed training set, sized: [%d, %d]' % (train_gradients.shape[0], train_gradients.shape[1]))
        #OutputLog().write('Processed test set, sized: [%d, %d]' % (test_gradients.shape[0], test_gradients.shape[1]))

        #if compress:

        #    compressed_data = lincompress(numpy.concatenate((train_gradients, test_gradients)).T).T

        #    print 'Compressed data size: {0}'.format(compressed_data.shape)

        #    train_gradients = compressed_data[:train_gradients.shape[0], :]
        #    test_gradients = compressed_data[train_gradients.shape[0]:, :]

        #    OutputLog().write('Compressed training set, sized: [%d, %d]' % (train_gradients.shape[0], train_gradients.shape[1]))
        #    OutputLog().write('Compressed test set, sized: [%d, %d]' % (test_gradients.shape[0], test_gradients.shape[1]))




        #svm_classifier = LinearSVC()

        #train_labels = numpy.arange(10)
        #for i in range(train_gradients.shape[0] / 10 - 1):
        #    train_labels = numpy.concatenate((train_labels, numpy.arange(10)))

        #test_labels = numpy.arange(10)
        #for i in range(test_gradients.shape[0] / 10 - 1):
        #    test_labels = numpy.concatenate((test_labels, numpy.arange(10)))

        #svm_classifier.fit(train_gradients, train_labels)

        #test_predictions = svm_classifier.predict(test_gradients)
        #train_predictions = svm_classifier.predict(train_gradients)