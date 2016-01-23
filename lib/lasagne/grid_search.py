import matplotlib
import scipy

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
from scipy.optimize import brute
from lib.MISC.container import Container
from lib.MISC.logger import OutputLog
from lib.MISC.utils import ConfigSectionMap, complete_rank, calculate_reconstruction_error, calculate_mardia
from lib.lasagne.Models import parallel_model, bisimilar_model, iterative_model, tied_dropout_iterative_model
from lib.lasagne.learnedactivations import batchnormalizeupdates
from lib.lasagne.params import Params
import lib.DataSetReaders

OUTPUT_DIR = r'C:\Workspace\output'


def iterate_minibatches(inputs_x, inputs_y, batchsize, shuffle=False):
    assert len(inputs_x) == len(inputs_y)
    if shuffle:
        indices = numpy.arange(len(inputs_x))
        numpy.random.shuffle(indices)
    for start_idx in range(0, len(inputs_x) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs_x[excerpt], inputs_y[excerpt]


def test_model(model_x, model_y, dataset_x, dataset_y, parallel=1):
    # Test
    y_values = model_x(dataset_x, dataset_y)
    x_values = model_y(dataset_x, dataset_y)

    OutputLog().write('\nTesting model\n')

    header = ['layer', 'loss', 'corr', 'search1', 'search5', 'search10', 'search_sum', 'desc1', 'desc5', 'desc10',
              'desc_sum']

    rows = []

    for index, (x, y) in enumerate(zip(x_values, y_values)):
        search_recall, describe_recall = complete_rank(x, y, data_set.reduce_val)
        loss = calculate_reconstruction_error(x, y)
        correlation = calculate_mardia(x, y, 0)

        print_row = ["{0} ".format(index), loss, correlation]
        print_row.extend(search_recall)
        print_row.append(sum(search_recall))
        print_row.extend(describe_recall)
        print_row.append(sum(describe_recall))

        rows.append(print_row)

    OutputLog().write(tabulate(rows, headers=header))


def update_param(param, value):
    if isinstance(param, list):
        for sub_param in param:
            update_param(sub_param, value)
    elif isinstance(param, tuple):
        Params.__dict__[param[0]][param[1]] = float(value)
        OutputLog().write('Param {0}[{1}] = {2}'.format(param[0], param[1], value))
    else:
        if isinstance(Params.__dict__[param], list):
            Params.__dict__[param] = [float(value) for i in Params.__dict__[param]]
            OutputLog().write('Param {0} = {1}'.format(param, value))
        else:
            Params.__dict__[param] = float(value)
            OutputLog().write('Param {0} = {1}'.format(param, value))


def fit(values, data_set, params):
    model = tied_dropout_iterative_model

    OutputLog().write('Model: {0}'.format(model.__name__))

    if len(params) == 1:
        update_param(params[0], values)
    else:
        for value, param in zip(values, params):
            update_param(param, value)

    model_x, model_y, hidden_x, hidden_y, loss, outputs, hooks = model.build_model(x_var,
                                                                                   data_set.trainset[0].shape[1],
                                                                                   y_var,
                                                                                   data_set.trainset[1].shape[1],
                                                                                   layer_sizes=Params.LAYER_SIZES,
                                                                                   parallel_width=Params.PARALLEL_WIDTH,
                                                                                   drop_prob=Params.DROPOUT,
                                                                                   weight_init=Params.WEIGHT_INIT)

    params_x = lasagne.layers.get_all_params(model_x, trainable=True)
    params_y = lasagne.layers.get_all_params(model_y, trainable=True)

    updates = OrderedDict(batchnormalizeupdates(hooks, 100))

    params_x.extend(params_y)

    params = lasagne.utils.unique(params_x)

    current_learning_rate = Params.BASE_LEARNING_RATE

    updates.update(
        lasagne.updates.momentum(loss, params, learning_rate=current_learning_rate, momentum=Params.MOMENTUM))

    train_fn = theano.function([x_var, y_var], [loss] + outputs.values(), updates=updates)

    test_y = theano.function([x_var, y_var],
                             [lasagne.layers.get_output(layer, moving_avg_hooks=hooks, deterministic=True) for layer in
                              hidden_x],
                             on_unused_input='ignore')
    test_x = theano.function([x_var, y_var],
                             [lasagne.layers.get_output(layer, moving_avg_hooks=hooks, deterministic=True) for layer in
                              hidden_y],
                             on_unused_input='ignore')

    batch_number = data_set.trainset[0].shape[0] / Params.BATCH_SIZE

    output_string = '{0}/{1} loss: {2} '
    output_string += ' '.join(['{0}:{{{1}}}'.format(key, index + 3) for index, key in enumerate(outputs.keys())])

    for epoch in range(Params.EPOCH_NUMBER):
        OutputLog().write('Epoch {0}'.format(epoch))
        for index, batch in enumerate(
                iterate_minibatches(data_set.trainset[0], data_set.trainset[1], Params.BATCH_SIZE, True)):
            input_x, input_y = batch
            train_loss = train_fn(input_x, input_y)
            OutputLog().write(output_string.format(index, batch_number, *train_loss))

    x_values = test_y(data_set.tuning[0], data_set.tuning[1])
    y_values = test_x(data_set.tuning[0], data_set.tuning[1])

    OutputLog().write('\nValidating model\n')

    for index, (x, y) in enumerate(zip(x_values, y_values)):
        search_recall, describe_recall = complete_rank(x, y, data_set.reduce_val)
        validation_loss = calculate_reconstruction_error(x, y)
        correlation = calculate_mardia(x, y, 0)

        OutputLog().write('Layer {0} - loss: {1}, correlation: {2}, recall: {3}'.format(index,
                                                                                        validation_loss,
                                                                                        correlation,
                                                                                        sum(search_recall) + sum(
                                                                                            describe_recall)))

    return sum(search_recall) + sum(describe_recall)


if __name__ == '__main__':
    data_set_config = sys.argv[1]

    OutputLog().set_path(OUTPUT_DIR)
    OutputLog().set_verbosity('info')

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)

    y_var = tensor.fmatrix()
    x_var = tensor.fmatrix()

    Params.print_params()

    Params.EPOCH_NUMBER = 10

    ranges = (slice(500, 4000, 500),)
    brute(fit, ranges, args=(data_set, [('LAYER_SIZES',2)]))