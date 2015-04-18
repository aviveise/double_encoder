import gc
from sklearn.linear_model import SGDClassifier
from MISC.whiten_transform import WhitenTransform
from stacked_auto_encoder_2 import StackedDoubleEncoder2

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
from MISC.utils import ConfigSectionMap, unitnorm_rows, center
from MISC.logger import OutputLog

import DataSetReaders
import Regularizations
import Optimizations


def compute_square(data_set, transformer):

    output = numpy.ndarray((data_set.trainset[0].shape[1] + data_set.testset[1].shape[1],
                            data_set.trainset[0].shape[1] + data_set.testset[1].shape[1]))

    for index_i, sample_i in enumerate(transformer.compute_outputs(data_set.trainset[0].T, data_set.trainset[1].T, 1)):

        for index_j, sample_j in enumerate(transformer.compute_outputs(data_set.trainset[0][:, index_i:].T,
                                                                       data_set.trainset[1][:, index_i:].T, 1)):

            output[index_i, index_j + index_i] = numpy.dot(sample_i, sample_j.reshape((sample_j.shape[0], 1)))
            output[index_j + index_i, index_i] = output[index_i, index_j + index_i]

        for index_j, sample_j in enumerate(transformer.compute_outputs(data_set.testset[0].T, data_set.testset[1].T, 1)):

            output[index_i, index_j + data_set.trainset[0].shape[1]] = numpy.dot(sample_i,
                                                                                   sample_j.reshape((sample_j.shape[0], 1)))

            output[index_j + data_set.trainset[0].shape[1], index_i] = output[index_i,
                                                                              index_j + data_set.trainset[0].shape[1]]

    for index_i, sample_i in enumerate(transformer.compute_outputs(data_set.testset[0].T, data_set.testset[1].T, 1)):
        for index_j, sample_j in enumerate(transformer.compute_outputs(data_set.testset[0].T, data_set.testset[1].T, 1)):
            output[index_i + data_set.trainset[0].shape[1], index_j + data_set.trainset[0].shape[1]] = \
                numpy.dot(sample_i, sample_j.reshape((sample_j.shape[0], 1)))

    return output


def lincompress(x):
    U, S, V = scipy.linalg.svd(numpy.dot(x, x.T))
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

    clf = SGDClassifier(alpha=0.005)
    samples = []
    labels = range(10)
    for epoch in range(configuration.hyper_parameters.epochs):
        for index, sample in enumerate(transformer.compute_outputs(data_set.trainset[0].T, data_set.trainset[1].T, 1)):

            samples.append(sample.reshape((1, sample.shape[0])))
            if index % 10 == 9:
                clf.partial_fit(samples, labels, labels)
                samples = []
                gc.collect()

    error = 0
    count = 0
    test_predictions = []
    for index, sample in enumerate(transformer.compute_outputs(data_set.testset[0].T, data_set.testset[1].T, 1)):
        prediction = clf.predict(sample)
        if not prediction == index % 10:
            error += 1

        count += 1
        test_predictions.append(prediction)

    OutputLog().write('test predictions weight: {0}'.format(test_predictions))

    OutputLog().write('\nerror: %f%%\n' % error)


def calc_gradient(gradient_file, layer=0):

    encoder = scipy.io.loadmat(gradient_file)

    layer_name = 'layer' + str(layer)
    next_layer_name = 'layer' + str(layer)

    wx_gradient = encoder['Wx_' + layer_name]

    if 'Wy_' + layer_name in encoder:
        wy_gradient = encoder['Wy_' + layer_name]
    else:
        wy_gradient = encoder['Wx_' + next_layer_name]

    return wx_gradient.flatten()
    #return numpy.concatenate((wx_gradient.flatten(), wy_gradient.flatten()))


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
        run_time_config = sys.argv[1]
        layer = int(sys.argv[2])

        #parse runtime configuration
        configuration = Configuration(run_time_config)

        gradient_train_path = os.path.join(configuration.output_parameters['path'], 'train')
        gradient_test_path = os.path.join(configuration.output_parameters['path'], 'test')

        # Selecting only probe files from the train gradient dir.
        gradient_train_files = [os.path.join(gradient_train_path, probe_file) for
                                probe_file in os.listdir(gradient_train_path)
                                if os.path.isfile(os.path.join(gradient_train_path, probe_file))]

        # Selecting only probe files from the test gradient dir.
        gradient_test_files = [os.path.join(gradient_test_path, probe_file) for
                               probe_file in os.listdir(gradient_test_path)
                               if os.path.isfile(os.path.join(gradient_test_path, probe_file))]

        x = numpy.zeros((len(gradient_train_files) + len(gradient_test_files),
                         len(gradient_train_files) + len(gradient_test_files)))

        sample_number = int(configuration.output_parameters['sample_number'])

        if bool(int(configuration.output_parameters['sample'])):

            print 'reading train:'
            x = None
            for train_file in gradient_train_files:
                fisher_vector = calc_gradient(train_file, layer)[range(sample_number)]
                fisher_vector -= numpy.mean(fisher_vector)
                fisher_vector /= numpy.linalg.norm(fisher_vector)
                file_name = os.path.split(os.path.splitext(train_file)[0])[1]
                sample = int(file_name.split('_')[-1])
                print sample
                if x is None:
                    x = numpy.ones((1800, fisher_vector.shape[0]))
                x[sample, :] = fisher_vector

            print 'reading test'
            for test_file in gradient_test_files:
                fisher_vector = calc_gradient(test_file, layer)[range(sample_number)]
                fisher_vector -= numpy.mean(fisher_vector)
                fisher_vector /= numpy.linalg.norm(fisher_vector)
                file_name = os.path.split(os.path.splitext(test_file)[0])[1]
                sample = int(file_name.split('_')[-1])
                print sample
                x[sample + 900, :] = fisher_vector

        else:

            x = None
            for train_file in gradient_train_files:
                fisher_vector = calc_gradient(train_file, layer)
                fisher_vector -= numpy.mean(fisher_vector)
                fisher_vector /= numpy.linalg.norm(fisher_vector)
                file_name = os.path.split(os.path.splitext(train_file)[0])[1]
                sample_number = int(file_name.split('_')[-1])
                if x is None:
                    x = numpy.zeros((1800, fisher_vector.shape[0]))
                x[sample_number, :] = fisher_vector

            for test_file in gradient_test_files:
                fisher_vector = calc_gradient(test_file, layer)
                fisher_vector -= numpy.mean(fisher_vector)
                fisher_vector /= numpy.linalg.norm(fisher_vector)
                file_name = os.path.split(os.path.splitext(test_file)[0])[1]
                sample_number = int(file_name.split('_')[-1])
                x[sample_number + 900, :] = fisher_vector

        #x = lincompress(x)

        train_gradients = x[0:900, :]#compressed_data[0:900, :]
        test_gradients = x[900:1800, :]#compressed_data[900:1800, :]

        w = WhitenTransform.fit(train_gradients.T)

        train_gradients = WhitenTransform.transform(train_gradients.T, w).T
        test_gradients = WhitenTransform.transform(test_gradients.T, w).T

        print 'train_gradient'
        print train_gradients

        print 'test_gradient'
        print test_gradients

        svm_classifier = LinearSVC()

        train_labels = numpy.arange(10)
        for i in range(train_gradients.shape[0] / 10 - 1):
           train_labels = numpy.concatenate((train_labels, numpy.arange(10)))

        test_labels = numpy.arange(10)
        for i in range(test_gradients.shape[0] / 10 - 1):
           test_labels = numpy.concatenate((test_labels, numpy.arange(10)))

        svm_classifier.fit(train_gradients, train_labels)

        test_predictions = svm_classifier.predict(test_gradients)
        train_predictions = svm_classifier.predict(train_gradients)

        OutputLog().write('test predictions:' + str(test_predictions))
        OutputLog().write('train predictions:' + str(train_predictions))

        error_test = float(numpy.count_nonzero(test_predictions - test_labels)) / test_labels.shape[0] * 100
        error_train = float(numpy.count_nonzero(train_predictions - train_labels)) / train_predictions.shape[0] * 100


        OutputLog().write('\nerror train: %f%%\n' % error_train)
        OutputLog().write('\nerror test: %f%%\n' % error_test)

                # for row_ndx, gradient_row_train_file in enumerate(gradient_train_files):
            #
            #     gradient_row_train = calc_gradient(gradient_row_train_file, layer)
            #
            #     #inserting into diagonal
            #     x[row_ndx, row_ndx] = numpy.dot(gradient_row_train,
            #                                     gradient_row_train.reshape((gradient_row_train.shape[0], 1)))
            #
            #     #inserting into row & col for train
            #     for col_ndx, gradient_col_train_file in enumerate(gradient_train_files[(row_ndx + 1):]):
            #
            #         gradient_col_train = calc_gradient(gradient_col_train_file, layer)
            #
            #         x[row_ndx, col_ndx + row_ndx + 1] = numpy.dot(gradient_row_train,
            #                                                       gradient_col_train.reshape((gradient_col_train.shape[0], 1)))
            #
            #         x[col_ndx + row_ndx + 1, row_ndx] = x[row_ndx, col_ndx + row_ndx + 1]
            #
            #     #inserting into row & col for test
            #     for col_ndx, gradient_col_test_file in enumerate(gradient_test_files):
            #
            #         gradient_col_test = calc_gradient(gradient_col_test_file, layer)
            #
            #         x[row_ndx, col_ndx + len(gradient_train_files)] = numpy.dot(gradient_row_train,
            #                                                                     gradient_col_test.reshape((gradient_col_test.shape[0], 1)))
            #
            #         x[col_ndx + len(gradient_train_files), row_ndx] = x[row_ndx, col_ndx + len(gradient_train_files)]
            #
            # for i_ndx, gradient_row_test_file in enumerate(gradient_test_files):
            #
            #     gradient_row_test = calc_gradient(gradient_row_test_file, layer)
            #
            #     row_ndx = i_ndx + len(gradient_test_files)
            #
            #     #inserting into diagonal
            #     x[row_ndx, row_ndx] = numpy.dot(gradient_row_test,
            #                                     gradient_row_test.reshape((gradient_row_test.shape[0], 1)))
            #
            #     #inserting into row & col for test
            #     for j_ndx, gradient_col_test_file in enumerate(gradient_test_files[i_ndx + 1:]):
            #
            #         gradient_col_test = calc_gradient(gradient_col_test_file, layer)
            #
            #         col_ndx = j_ndx + len(gradient_test_files)
            #
            #         x[row_ndx, col_ndx + row_ndx + 1] = numpy.dot(gradient_row_test,
            #                                              gradient_col_test.reshape((gradient_col_test.shape[0], 1)))
            #
            #         x[col_ndx + row_ndx + 1, row_ndx] = x[row_ndx, col_ndx + row_ndx + 1]


    # @staticmethod
    # def run_old():
    #
    #     data_set_config = sys.argv[1]
    #     run_time_config = sys.argv[2]
    #     double_encoder = sys.argv[3]
    #     network_type = sys.argv[4]
    #     type = sys.argv[5]
    #     layer = int(sys.argv[6])
    #
    #     data_config = ConfigParser.ConfigParser()
    #     data_config.read(data_set_config)
    #     data_parameters = ConfigSectionMap("dataset_parameters", data_config)
    #
    #     #construct data set
    #     data_set = Container().create(data_parameters['name'], data_parameters)
    #
    #     #parse runtime configuration
    #     configuration = Configuration(run_time_config)
    #     configuration.hyper_parameters.batch_size = int(configuration.hyper_parameters.batch_size * data_set.trainset[0].shape[1])
    #
    #     training_set_x = data_set.trainset[0].T
    #     training_set_y = data_set.trainset[1].T
    #
    #     if network_type == 'TypeA':
    #         symmetric_double_encoder = StackedDoubleEncoder(hidden_layers=[],
    #                                                         numpy_range=RandomState(),
    #                                                         input_size_x=training_set_x.shape[1],
    #                                                         input_size_y=training_set_y.shape[1],
    #                                                         batch_size=configuration.hyper_parameters.batch_size,
    #                                                         activation_method=None)
    #     else:
    #         symmetric_double_encoder = StackedDoubleEncoder2(hidden_layers=[],
    #                                                          numpy_range=RandomState(),
    #                                                          input_size_x=training_set_x.shape[1],
    #                                                          input_size_y=training_set_y.shape[1],
    #                                                          batch_size=configuration.hyper_parameters.batch_size,
    #                                                          activation_method=None)
    #
    #     symmetric_double_encoder.import_encoder(double_encoder, configuration.hyper_parameters)
    #
    #     OutputLog().write('calculating gradients')
    #
    #     params_x = [symmetric_double_encoder[layer].Wx]
    #     params_y = [symmetric_double_encoder[layer].Wy]
    #     params = [symmetric_double_encoder[layer].Wy, symmetric_double_encoder[layer].Wx]
    #     params_var = [symmetric_double_encoder.var_x, symmetric_double_encoder.var_y]
    #
    #     print symmetric_double_encoder[layer].Wx.get_value(borrow=True).shape
    #
    #     transformer_x = GradientTransformer(symmetric_double_encoder, params_x, configuration.hyper_parameters)
    #     transformer_y = GradientTransformer(symmetric_double_encoder, params_y, configuration.hyper_parameters)
    #     transformer = GradientTransformer(symmetric_double_encoder, params, configuration.hyper_parameters)
    #     transformer_var = GradientTransformer(symmetric_double_encoder, params_var, configuration.hyper_parameters)
    #
    #     if type == 'SGD':
    #         print 'Wx:'
    #         test_transformer(transformer_x, data_set, configuration)
    #
    #         print 'Wy:'
    #         test_transformer(transformer_y, data_set, configuration)
    #
    #         print 'Wx+Wy:'
    #         test_transformer(transformer, data_set, configuration)
    #
    #         print 'var:'
    #         test_transformer(transformer_var, data_set, configuration)
    #
    #     else:
    #
    #         x = compute_square(data_set, transformer)
    #
    #         compressed_data = lincompress(x)
    #
    #         train_gradients = compressed_data[:data_set.trainset.shape[1], :]
    #         test_gradients = compressed_data[data_set.trainset.shape[1]:, :]
    #
    #         svm_classifier = LinearSVC()
    #
    #         train_labels = numpy.arange(10)
    #         for i in range(train_gradients.shape[0] / 10 - 1):
    #            train_labels = numpy.concatenate((train_labels, numpy.arange(10)))
    #
    #         test_labels = numpy.arange(10)
    #         for i in range(test_gradients.shape[0] / 10 - 1):
    #            test_labels = numpy.concatenate((test_labels, numpy.arange(10)))
    #
    #         svm_classifier.fit(train_gradients, train_labels)
    #
    #         test_predictions = svm_classifier.predict(test_gradients)
    #         train_predictions = svm_classifier.predict(train_gradients)
    #
    #         OutputLog().write('test predictions:' + str(test_predictions))
    #
    #         error_test = float(numpy.count_nonzero(test_predictions - test_labels)) / test_labels.shape[0] * 100
    #         error_train = float(numpy.count_nonzero(train_predictions - train_labels)) / train_predictions.shape[0] * 100
    #
    #
    #         OutputLog().write('\nerror train: %f%%\n' % error_train)
    #         OutputLog().write('\nerror test: %f%%\n' % error_test)