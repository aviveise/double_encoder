from math import floor

import matplotlib

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
from lib.MISC.utils import ConfigSectionMap, complete_rank, calculate_reconstruction_error, calculate_mardia
from lib.lasagne.Models import parallel_model, bisimilar_model, iterative_model, tied_dropout_iterative_model
from lib.lasagne.learnedactivations import batchnormalizeupdates
from lib.lasagne.params import Params
import lib.DataSetReaders

OUTPUT_DIR = r'C:\Workspace\output'
VALIDATE_ALL = False


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


def test_model(model_x, model_y, dataset_x, dataset_y, parallel=1, validate_all=True):
    # Test
    y_values = model_x(dataset_x, dataset_y)
    x_values = model_y(dataset_x, dataset_y)

    OutputLog().write('\nTesting model\n')

    header = ['layer', 'loss', 'corr', 'search1', 'search5', 'search10', 'search_sum', 'desc1', 'desc5', 'desc10',
              'desc_sum']

    rows = []

    if validate_all:
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
    else:
        middle_x = x_values[Params.TEST_LAYER]
        middle_y = y_values[Params.TEST_LAYER]

        search_recall, describe_recall = complete_rank(middle_x, middle_y, data_set.reduce_val)
        loss = calculate_reconstruction_error(middle_x, middle_y)
        correlation = calculate_mardia(middle_x, middle_y, 0)

        print_row = ["{0} ".format(middle), loss, correlation]
        print_row.extend(search_recall)
        print_row.append(sum(search_recall))
        print_row.extend(describe_recall)
        print_row.append(sum(describe_recall))

        rows.append(print_row)

    OutputLog().write(tabulate(rows, headers=header))


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

    model = tied_dropout_iterative_model

    Params.print_params()

    OutputLog().write('Model: {0}'.format(model.__name__))

    # Export network
    path = OutputLog().output_path

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
            train_loss = train_fn(numpy.cast[theano.config.floatX](input_x),
                                  numpy.cast[theano.config.floatX](input_y))
            OutputLog().write(output_string.format(index, batch_number, *train_loss))

        x_values = test_y(data_set.tuning[0], data_set.tuning[1])
        y_values = test_x(data_set.tuning[0], data_set.tuning[1])

        OutputLog().write('\nValidating model\n')

        if VALIDATE_ALL:
            for index, (x, y) in enumerate(zip(x_values, y_values)):
                search_recall, describe_recall = complete_rank(x, y, data_set.reduce_val)
                validation_loss = calculate_reconstruction_error(x, y)
                correlation = calculate_mardia(x, y, 0)

                OutputLog().write('Layer {0} - loss: {1}, correlation: {2}, recall: {3}'.format(index,
                                                                                                validation_loss,
                                                                                                correlation,
                                                                                                sum(
                                                                                                    search_recall) + sum(
                                                                                                    describe_recall)))
        else:
            middle = int(len(x_values) / 2.) - 1 if len(x_values) % 2 == 0 else int(floor(float(len(x_values)) / 2.))
            middle_x = x_values[middle]
            middle_y = y_values[middle]
            search_recall, describe_recall = complete_rank(middle_x, middle_y, data_set.reduce_val)
            validation_loss = calculate_reconstruction_error(middle_x, middle_y)
            correlation = calculate_mardia(middle_x, middle_y, 0)

            OutputLog().write('Layer - loss: {1}, correlation: {2}, recall: {3}'.format(index,
                                                                                        validation_loss,
                                                                                        correlation,
                                                                                        sum(search_recall) + sum(
                                                                                            describe_recall)))

        del x_values
        del y_values

        if epoch in Params.DECAY_EPOCH:
            current_learning_rate *= Params.DECAY_RATE
            updates = OrderedDict(batchnormalizeupdates(hooks, 100))
            test_model(test_x, test_y, data_set.testset[0], data_set.testset[1], parallel=5, validate_all=VALIDATE_ALL)

            with file(os.path.join(path, 'model_x_{0}.p'.format(epoch)), 'w') as model_x_file:
                cPickle.dump(model_x, model_x_file)

            with file(os.path.join(path, 'model_y{0}.p'.format(epoch)), 'w') as model_y_file:
                cPickle.dump(model_y, model_y_file)

            updates.update(
                lasagne.updates.nesterov_momentum(loss, params, learning_rate=current_learning_rate, momentum=0.9))
            del train_fn
            train_fn = theano.function([x_var, y_var], [loss] + outputs.values(), updates=updates)

    OutputLog().write('Test results')

    test_model(test_x, test_y, data_set.testset[0], data_set.testset[1], parallel=5)

    with file(os.path.join(path, 'model_x.p'), 'w') as model_x_file:
        cPickle.dump(model_x, model_x_file)

    with file(os.path.join(path, 'model_y.p'), 'w') as model_y_file:
        cPickle.dump(model_y, model_y_file)
