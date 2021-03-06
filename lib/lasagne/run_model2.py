import traceback
from copy import copy
from math import floor
import matplotlib
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


def test_model(model_x, model_y, dataset_x, dataset_y, parallel=1, validate_all=True, top=0, x_y_mapping=None,
               x_reduce=None):
    # Testing
    test_x, test_y = preprocess_dataset_test(dataset_x, dataset_y, x_reduce)
    # test_x = dataset_x
    # test_y = dataset_y

    if dataset_x.shape[0] > 10000:
        validate_all = False
        x_total_value = None
        y_total_value = None
        for index, batch in enumerate(
                iterate_minibatches(test_x, test_y, Params.VALIDATION_BATCH_SIZE, True)):
            input_x, input_y = batch
            y_values = model_x(input_x, input_y)[Params.TEST_LAYER]
            x_values = model_y(input_x, input_y)[Params.TEST_LAYER]

            if not x_total_value:
                x_total_value = x_values
            else:
                x_total_value = numpy.vstack((x_total_value, x_values))

            if not y_total_value:
                y_total_value = y_values
            else:
                y_total_value = numpy.vstack((y_total_value, y_values))
    else:
        y_values = model_x(test_x, test_y)
        x_values = model_y(test_x, test_y)

        if not validate_all:
            x_total_value = x_values[Params.TEST_LAYER]
            y_total_value = y_values[Params.TEST_LAYER]

    OutputLog().write('\nTesting model\n')

    header = ['layer', 'loss', 'corr', 'search1', 'search5', 'search10', 'search_sum', 'desc1', 'desc5', 'desc10',
              'desc_sum', 'mrr', 'map']

    rows = []

    if validate_all:
        for index, (x, y) in enumerate(zip(x_values, y_values)):
            if data_set.x_y_mapping is not None:
                similarity = compute_similarity(x, y, x_y_mapping)
                search_recall, describe_recall, mrr, map = complete_rank_2(x, y, x_y_mapping, x_reduce, similarity)
            else:
                search_recall, describe_recall = complete_rank(x, y, data_set.reduce_val)
                mrr = 0
                map = 0

            loss = calculate_reconstruction_error(x, y)
            correlation = calculate_mardia(x, y, top)

            print_row = ["{0} ".format(index), loss, correlation]
            print_row.extend(search_recall)
            print_row.append(sum(search_recall))
            print_row.extend(describe_recall)
            print_row.append(sum(describe_recall))
            print_row.append(mrr)
            print_row.append(map)

            rows.append(print_row)
    else:
        middle_x = x_total_value
        middle_y = y_total_value

        if data_set.x_y_mapping is not None:
            similarity = compute_similarity(middle_x, middle_y, x_y_mapping)
            search_recall, describe_recall, mrr, map = complete_rank_2(middle_x, middle_y, x_y_mapping, x_reduce, similarity)
        else:
            search_recall, describe_recall = complete_rank(middle_x, middle_y, data_set.reduce_val)
            mrr = 0
            map = 0

        loss = calculate_reconstruction_error(middle_x, middle_y)
        correlation = calculate_mardia(middle_x, middle_y, top)

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


def preprocess_dataset_train(train_x, train_y, reduce_x, x_y_mapping):
    # x_r = train_x[reduce_x]
    # result_x = numpy.zeros((x_r.shape[0], train_y.shape[1]), dtype=theano.config.floatX)
    # result_y = numpy.zeros((x_r.shape[0], train_y.shape[1]), dtype=theano.config.floatX)

    result_x = numpy.zeros((train_x.shape[0], train_y.shape[1]), dtype=theano.config.floatX)
    result_y = numpy.zeros((train_x.shape[0], train_y.shape[1]), dtype=theano.config.floatX)

    y_indices = copy(reduce_x)
    y_indices.append(train_y.shape[0])
    # for index, x in enumerate(x_r):
    #     result_x[index] = x
    #     result_y[index] = numpy.mean(train_y[y_indices[index]: y_indices[index + 1]], axis=0)
    #     result_y[index] = result_y[index] / numpy.var(result_y[index])
    q_index = -1
    for index, (x, y) in enumerate(zip(train_x, train_y)):
        if index in reduce_x:
            q_index += 1
        label = x_y_mapping[q_index, index]
        result_x[index, :] = x
        result_y[index, :] = y


    return result_x, result_y


def compute_similarity(middle_x, middle_y, x_y_mapping):
    scores = numpy.zeros(middle_x.shape[0])
    for index, (x, y) in enumerate(zip(middle_x, middle_y)):
        scores[index] = cdist(numpy.reshape(x, [1, x.shape[0]]),
                              numpy.reshape(y, [1, y.shape[0]]), metric=Params.SIMILARITY_METRIC)

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

    OutputLog().set_path(OUTPUT_DIR)
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

    x_train, y_train = preprocess_dataset_train(data_set.trainset[0], data_set.trainset[1], data_set.x_reduce['train'],
                                                data_set.x_y_mapping['train'])
    #
    # x_train = data_set.trainset[0]
    # y_train = data_set.trainset[1]

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
        # lasagne.updates.momentum(loss, params, learning_rate=current_learning_rate, momentum=Params.MOMENTUM))

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

        model_results['train'].append({'loss': []})
        model_results['validate'].append({})

        for label in outputs.keys():
            model_results['train'][epoch][label] = []

        for index, batch in enumerate(
                iterate_minibatches(x_train, y_train, Params.BATCH_SIZE, True)):
            input_x, input_y = batch
            train_loss = train_fn(numpy.cast[theano.config.floatX](input_x),
                                  numpy.cast[theano.config.floatX](input_y))

            model_results['train'][epoch]['loss'].append(train_loss[0])
            for label, value in zip(outputs.keys(), train_loss[1:]):
                model_results['train'][epoch][label].append(value)

            OutputLog().write(output_string.format(index, batch_number, *train_loss))

        if Params.CROSS_VALIDATION:
            tuning_x, tuning_y = preprocess_dataset_test(data_set.tuning[0], data_set.tuning[1],
                                                        data_set.x_reduce['dev'])

            # tuning_x = data_set.tuning[0]
            # tuning_y = data_set.tuning[1]

            x_values = test_y(tuning_x, tuning_y)
            y_values = test_x(tuning_x, tuning_y)

            OutputLog().write('\nValidating model\n')

            if VALIDATE_ALL:
                for index, (x, y) in enumerate(zip(x_values, y_values)):
                    if data_set.x_y_mapping['dev'] is not None:
                        search_recall, describe_recall, mrr, map = complete_rank_2(x, y, data_set.x_y_mapping['dev'],
                                                                                   data_set.x_reduce['dev'])
                    else:
                        search_recall, describe_recall = complete_rank(x, y, data_set.reduce_val)
                        mrr = 0
                        map = 0

                    validation_loss = calculate_reconstruction_error(x, y)
                    correlation = calculate_mardia(x, y, top)

                    OutputLog().write(
                        'Layer {0} - loss: {1}, correlation: {2}, recall: {3}, mrr: {4}, map: {5}'.format(index,
                                                                                                          validation_loss,
                                                                                                          correlation,
                                                                                                          sum(
                                                                                                              search_recall) + sum(
                                                                                                              describe_recall),
                                                                                                          mrr, map))
            else:
                middle_x = x_values[Params.TEST_LAYER]
                middle_y = y_values[Params.TEST_LAYER]

                if data_set.x_y_mapping['dev'] is not None:
                    similarity = compute_similarity(middle_x, middle_y, data_set.x_y_mapping['dev'])
                    search_recall, describe_recall, mrr, map = complete_rank_2(middle_x, middle_y,
                                                                               data_set.x_y_mapping['dev'],
                                                                               data_set.x_reduce['dev'],
                                                                               similarity)
                else:
                    search_recall, describe_recall = complete_rank(middle_x, middle_y, data_set.reduce_val)
                    mrr = 0
                    map = 0

                validation_loss = calculate_reconstruction_error(middle_x, middle_y)
                correlation = calculate_mardia(middle_x, middle_y, top)
                mean_x = numpy.mean(numpy.mean(middle_x, axis=0)),
                mean_y = numpy.mean(numpy.mean(middle_y, axis=0)),
                var_x = numpy.mean(numpy.var(middle_x, axis=0)),
                var_y = numpy.mean(numpy.var(middle_y, axis=0)),

                OutputLog().write('Layer - loss: {1}, correlation: {2}, recall: {3}, mean_x: {4}, mean_y: {5},'
                                  'var_x: {6}, var_y: {7}, mrr: {8}, map: {9}'.format(index,
                                                                                      validation_loss,
                                                                                      correlation,
                                                                                      sum(search_recall) + sum(
                                                                                          describe_recall),
                                                                                      mean_x, mean_y, var_x, var_y, mrr,
                                                                                      map))

                model_results['validate'][epoch]['loss'] = validation_loss
                model_results['validate'][epoch]['correlation'] = correlation
                model_results['validate'][epoch]['search_recall'] = sum(search_recall)
                model_results['validate'][epoch]['describe_recall'] = sum(describe_recall)
                model_results['validate'][epoch]['mean_x'] = mean_x
                model_results['validate'][epoch]['mean_y'] = mean_y
                model_results['validate'][epoch]['var_x'] = var_x
                model_results['validate'][epoch]['var_y'] = var_y

        if epoch in Params.DECAY_EPOCH:
            current_learning_rate *= Params.DECAY_RATE
            if hooks:
                updates = OrderedDict(batchnormalizeupdates(hooks, 100))
            else:
                updates = OrderedDict()
            try:
                test_model(test_x, test_y, numpy.cast[theano.config.floatX](data_set.testset[0]),
                           numpy.cast[theano.config.floatX](data_set.testset[1]),
                           top=top, x_y_mapping=data_set.x_y_mapping['test'], x_reduce=data_set.x_reduce['test'])

            except Exception as e:
                OutputLog().write('Failed testing model with exception {0}'.format(e))
                OutputLog().write('{0}'.format(traceback.format_exc()))

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
                   x_reduce=data_set.x_reduce['test'])
    except Exception as e:
        OutputLog().write('Error testing model with exception {0}'.format(e))
        traceback.print_exc()

    with file(os.path.join(path, 'model_x.p'), 'w') as model_x_file:
        cPickle.dump(model_x, model_x_file)

    with file(os.path.join(path, 'model_y.p'), 'w') as model_y_file:
        cPickle.dump(model_y, model_y_file)

    with file(os.path.join(path, 'results.p'), 'w') as results_file:
        cPickle.dump(model_results, results_file)
