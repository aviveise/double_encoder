from math import floor
import matplotlib
import traceback
from scipy.spatial.distance import cdist
from sklearn import preprocessing

matplotlib.use('Agg')

import ConfigParser
import os
import sys
import cPickle
import lasagne
import numpy
import json
from collections import OrderedDict
from tabulate import tabulate
from theano import tensor, theano
from lib.MISC.container import Container
from lib.MISC.logger import OutputLog
from lib.MISC.utils import ConfigSectionMap, complete_rank, calculate_reconstruction_error, calculate_mardia, \
    complete_rank_2, scale_cols
from lib.lasagne.Models import parallel_model, bisimilar_model, iterative_model, tied_dropout_iterative_model
from lib.lasagne.learnedactivations import batchnormalizeupdates
from lib.lasagne.params import Params
import lib.DataSetReaders

OUTPUT_DIR = r'/specific/a/netapp3/vol/wolf/davidgad/aviveise/results/'
VALIDATE_ALL = False


def iterate_parallel_minibatches(inputs_x, inputs_y, batchsize, shuffle=False, preprocessors=None):
    assert len(inputs_x) == len(inputs_y)
    if shuffle:
        indices = numpy.arange(len(inputs_x))
        numpy.random.shuffle(indices)
    for start_idx in range(0, len(inputs_x) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if preprocessors is not None:
            yield preprocessors[0](numpy.copy(inputs_x[excerpt])), preprocessors[1](numpy.copy(inputs_y[excerpt]))
        else:
            yield inputs_x[excerpt], inputs_y[excerpt]

def iterate_single_minibatch(inputs, batchsize, shuffle=False, preprocessor=None):
    if shuffle:
        indices = numpy.arange(len(inputs))
        numpy.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if preprocessor is not None:
            yield preprocessor(numpy.copy(inputs[excerpt]))
        else:
            yield inputs[excerpt]


def test_model(model_x, model_y, dataset_x, dataset_y, parallel=1, validate_all=True, top=0, x_y_mapping=None,
               x_reduce=None, preprocessors=None):
    test_x = dataset_x
    test_y = dataset_y

    x_total_value = None
    y_total_value = None
    for index, batch in enumerate(
            iterate_single_minibatch(test_x, Params.VALIDATION_BATCH_SIZE, False, preprocessor=preprocessors[0])):
        x_values = model_y(batch)[Params.TEST_LAYER]

        if x_total_value is None:
            x_total_value = x_values
        else:
            x_total_value = numpy.vstack((x_total_value, x_values))

    for index, batch in enumerate(
            iterate_single_minibatch(test_y, Params.VALIDATION_BATCH_SIZE, False, preprocessor=preprocessors[1])):

        y_values = model_x(batch)[Params.TEST_LAYER]

        if y_total_value is None:
            y_total_value = y_values
        else:
            y_total_value = numpy.vstack((y_total_value, y_values))


    header = ['layer', 'loss', 'corr', 'search1', 'search5', 'search10', 'search_sum', 'desc1', 'desc5', 'desc10',
              'desc_sum', 'mrr', 'map']

    rows = []

    if x_y_mapping is not None:
        search_recall, describe_recall, mrr, map = complete_rank_2(x_total_value, y_total_value, x_y_mapping, x_reduce, None)
    else:
        search_recall, describe_recall = complete_rank(x_total_value, y_total_value, data_set.reduce_val)
        mrr = 0
        map = 0

    loss = calculate_reconstruction_error(x_total_value, y_total_value)
    correlation = calculate_mardia(x_total_value, y_total_value, top)

    print_row = ["{0} ".format(Params.TEST_LAYER), loss, correlation]
    print_row.extend(search_recall)
    print_row.append(sum(search_recall))
    print_row.extend(describe_recall)
    print_row.append(sum(describe_recall))
    print_row.append(mrr)
    print_row.append(map)

    rows.append(print_row)

    OutputLog().write(tabulate(rows, headers=header))


def preprocess_dataset_test(test_x, test_y, reduce_x):
    reduced_x = test_x[reduce_x]

    result_x = numpy.zeros((reduced_x.shape[0] * test_y.shape[0], test_y.shape[1]), dtype=theano.config.floatX)
    result_y = numpy.zeros((reduced_x.shape[0] * test_y.shape[0], test_y.shape[1]), dtype=theano.config.floatX)

    for i, x in enumerate(reduced_x):
        for j, y in enumerate(test_y):
            result_x[i * test_y.shape[0] + j, :] = x
            result_y[i * test_y.shape[0] + j, :] = y

    return result_x, result_y


def preprocess_dataset_train(train_x, train_y, reduce_x):
    x_r = train_x[reduce_x]
    result_x = numpy.zeros((x_r.shape[0], train_y.shape[1]), dtype=theano.config.floatX)
    result_y = numpy.zeros((x_r.shape[0], train_y.shape[1]), dtype=theano.config.floatX)

    y_indices = reduce_x
    y_indices.append(train_y.shape[0])
    for index, x in enumerate(x_r):
        result_x[index] = x
        result_y[index] = numpy.mean(train_y[y_indices[index]: y_indices[index + 1]], axis=0)
        result_y[index] = result_y[index] / numpy.var(result_y[index])

    return result_x, result_y


def compute_similarity(middle_x, middle_y, x_y_mapping, reduce_x):
    scores = numpy.zeros(x_y_mapping.shape[0] * x_y_mapping.shape[1])
    for index_x, x in enumerate(middle_x):
        for index_y, y in enumerate(middle_y):
            scores[index_x * middle_x.shape[0] + index_y] = cdist(numpy.reshape(x, [1, x.shape[0]]),
                                                                  numpy.reshape(y, [1, y.shape[0]]),
                                                                  metric=Params.SIMILARITY_METRIC)

    similarity = numpy.zeros(x_y_mapping.shape)

    for q_index in range(x_y_mapping.shape[0]):
        similarity[q_index, :] = scores[q_index * x_y_mapping.shape[1]: (q_index + 1) * x_y_mapping.shape[1]]

    return similarity


if __name__ == '__main__':

    data_set_config = sys.argv[1]
    if len(sys.argv) > 2:
        top = int(sys.argv[2])
    else:
        top = 0

    model_results = {'train': [], 'validate': []}

    results_folder = os.path.join(OUTPUT_DIR, str.rstrip(str.split(data_set_config,'_')[-1][:-4]))

    OutputLog().set_path(results_folder)
    OutputLog().set_verbosity('info')

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)
    data_set.load()

    y_var = tensor.fmatrix()
    x_var = tensor.fmatrix()

    model = tied_dropout_iterative_model

    Params.print_params()

    OutputLog().write('Model: {0}'.format(model.__name__))

    # Export network
    path = OutputLog().output_path

    x_train = data_set.trainset[0]
    y_train = data_set.trainset[1]

    model_x, model_y, hidden_x, hidden_y, loss, outputs, hooks = model.build_model(x_var,
                                                                                   x_train.shape[1],
                                                                                   y_var,
                                                                                   y_train.shape[1],
                                                                                   layer_sizes=Params.LAYER_SIZES,
                                                                                   parallel_width=Params.PARALLEL_WIDTH,
                                                                                   drop_prob=Params.DROPOUT,
                                                                                   weight_init=Params.WEIGHT_INIT)

    params_x = lasagne.layers.get_all_params(model_x, trainable=True)
    params_y = lasagne.layers.get_all_params(model_y, trainable=True)

    if hooks:
        updates = OrderedDict(batchnormalizeupdates(hooks, 100))
    else:
        updates = OrderedDict()

    params_x.extend(params_y)

    params = lasagne.utils.unique(params_x)

    current_learning_rate = Params.BASE_LEARNING_RATE

    updates.update(
        lasagne.updates.adam(loss, params, learning_rate=current_learning_rate))

    train_fn = theano.function([x_var, y_var], [loss] + outputs.values(), updates=updates)

    test_y = theano.function([x_var],
                             [lasagne.layers.get_output(layer, moving_avg_hooks=hooks, deterministic=True) for layer in
                              hidden_x],
                             on_unused_input='ignore')
    test_x = theano.function([y_var],
                             [lasagne.layers.get_output(layer, moving_avg_hooks=hooks, deterministic=True) for layer in
                              hidden_y],
                             on_unused_input='ignore')

    batch_number = data_set.trainset[0].shape[0] / Params.BATCH_SIZE

    output_string = '{0}/{1} loss: {2} '
    output_string += ' '.join(['{0}:{{{1}}}'.format(key, index + 3) for index, key in enumerate(outputs.keys())])

    for epoch in range(Params.EPOCH_NUMBER):
        OutputLog().write('Epoch {0}'.format(epoch))

        model_results['train'].append({'loss': []})
        model_results['validate'].append({})

        for label in outputs.keys():
            model_results['train'][epoch][label] = []

        for index, batch in enumerate(
                iterate_parallel_minibatches(x_train, y_train, Params.BATCH_SIZE, True, data_set.preprocessors)):
            input_x, input_y = batch
            train_loss = train_fn(numpy.cast[theano.config.floatX](input_x),
                                  numpy.cast[theano.config.floatX](input_y))

            model_results['train'][epoch]['loss'].append(train_loss[0])
            for label, value in zip(outputs.keys(), train_loss[1:]):
                model_results['train'][epoch][label].append(value)

            OutputLog().write(output_string.format(index, batch_number, *train_loss))

        if Params.CROSS_VALIDATION or epoch in Params.DECAY_EPOCH:
            tuning_x = data_set.tuning[0]
            tuning_y = data_set.tuning[1]

            OutputLog().write('\nValidating model\n')

            test_model(test_x, test_y, tuning_x, tuning_y, x_y_mapping=data_set.x_y_mapping['dev'],
                       x_reduce=data_set.x_reduce['dev'], preprocessors=data_set.preprocessors)

        if epoch in Params.DECAY_EPOCH:
            current_learning_rate *= Params.DECAY_RATE
            if hooks:
                updates = OrderedDict(batchnormalizeupdates(hooks, 100))
            else:
                updates = OrderedDict()

            with file(os.path.join(path, 'model_x_{0}.p'.format(epoch)), 'w') as model_x_file:
                cPickle.dump(model_x, model_x_file)

            with file(os.path.join(path, 'model_y{0}.p'.format(epoch)), 'w') as model_y_file:
                cPickle.dump(model_y, model_y_file)

            updates.update(
                lasagne.updates.nesterov_momentum(loss, params, learning_rate=current_learning_rate, momentum=0.9))
            del train_fn
            train_fn = theano.function([x_var, y_var], [loss] + outputs.values(), updates=updates)

    OutputLog().write('Test results')

    try:
        test_model(test_x, test_y, data_set.testset[0], data_set.testset[1], parallel=5, top=top,
                   x_y_mapping=data_set.x_y_mapping['test'],
                   x_reduce=data_set.x_reduce['test'],
                   preprocessors=data_set.preprocessors)
    except Exception as e:
        OutputLog().write('Error testing model with exception {0}'.format(e))
        traceback.print_exc()

    with file(os.path.join(path, 'model_x.p'), 'w') as model_x_file:
        cPickle.dump(model_x, model_x_file)

    with file(os.path.join(path, 'model_y.p'), 'w') as model_y_file:
        cPickle.dump(model_y, model_y_file)

    with file(os.path.join(path, 'results.p'), 'w') as results_file:
        cPickle.dump(model_results, results_file)
