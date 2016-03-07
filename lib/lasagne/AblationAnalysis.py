import ConfigParser
import copy
import os
import pickle
import sys
import traceback
import uuid
from collections import OrderedDict
import itertools
from multiprocessing import Pool
import functools
from uuid import UUID
import lasagne
import multiprocessing
import numpy
import simplejson
from tabulate import tabulate
from theano import tensor, theano
from lib.MISC.container import Container
from lib.MISC.logger import OutputLog
from lib.MISC.utils import ConfigSectionMap, complete_rank
from lib.MISC.utils import calculate_reconstruction_error, calculate_mardia
from lib.lasagne.experiments import experiments
from lib.lasagne.Layers import TiedDropoutLayer
from lib.lasagne.Models import tied_dropout_iterative_model
from lib.lasagne.learnedactivations import batchnormalizeupdates
from lib.lasagne.params import Params
import lib.DataSetReaders


OUTPUT_DIR = r'C:\Workspace\output'
VALIDATE_ALL = False
PROCESS_NUMBER = 4


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


def update_param(param):
    if isinstance(param, list):
        for sub_param in param:
            update_param(sub_param)
    elif isinstance(param, tuple):
        if isinstance(param[1], numpy.ndarray):
            Params.__dict__[param[0]] = [float(p) for p in param[1]]
        else:
            Params.__dict__[param[0]] = param[1]
        OutputLog().write('Param {0} = {1}'.format(param[0], param[1]))


def run_experiment(experiment_values, data_parameters, path):
    id = uuid.uuid4()
    OutputLog().set_output_path(path, suffix=str(id))

    top = 0

    param_backup = copy.deepcopy(Params.__dict__)
    update_param(experiment_values)

    y_var = tensor.fmatrix()
    x_var = tensor.fmatrix()

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)
    data_set.load()

    model_results = {'train': [], 'validate': []}

    model = tied_dropout_iterative_model

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

    if hooks:
        updates = OrderedDict(batchnormalizeupdates(hooks, 100))
    else:
        updates = OrderedDict()

    current_learning_rate = Params.BASE_LEARNING_RATE

    params_x.extend(params_y)
    params = lasagne.utils.unique(params_x)

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

        model_results['train'].append({'loss': []})
        model_results['validate'].append({})

        for label in outputs.keys():
            model_results['train'][epoch][label] = []

        for index, batch in enumerate(
                iterate_minibatches(data_set.trainset[0], data_set.trainset[1], Params.BATCH_SIZE, True)):
            input_x, input_y = batch
            train_loss = train_fn(numpy.cast[theano.config.floatX](input_x),
                                  numpy.cast[theano.config.floatX](input_y))

            model_results['train'][epoch]['loss'].append(train_loss[0])
            for label, value in zip(outputs.keys(), train_loss[1:]):
                model_results['train'][epoch][label].append(value)

            OutputLog().write(output_string.format(index, batch_number, *train_loss))

        if Params.CROSS_VALIDATION:
            x_values = test_y(data_set.tuning[0], data_set.tuning[1])
            y_values = test_x(data_set.tuning[0], data_set.tuning[1])

            OutputLog().write('\nValidating model\n')

            if VALIDATE_ALL:
                for index, (x, y) in enumerate(zip(x_values, y_values)):
                    search_recall, describe_recall = complete_rank(x, y, data_set.reduce_val)
                    validation_loss = calculate_reconstruction_error(x, y)
                    correlation = calculate_mardia(x, y, top)

                    OutputLog().write('Layer {0} - loss: {1}, correlation: {2}, recall: {3}'.format(index,
                                                                                                    validation_loss,
                                                                                                    correlation,
                                                                                                    sum(
                                                                                                        search_recall) + sum(
                                                                                                        describe_recall)))
            else:
                middle_x = x_values[Params.TEST_LAYER]
                middle_y = y_values[Params.TEST_LAYER]
                search_recall, describe_recall = complete_rank(middle_x, middle_y, data_set.reduce_val)
                validation_loss = calculate_reconstruction_error(middle_x, middle_y)
                correlation = calculate_mardia(middle_x, middle_y, top)
                mean_x = numpy.mean(numpy.mean(middle_x, axis=0)),
                mean_y = numpy.mean(numpy.mean(middle_y, axis=0)),
                var_x = numpy.mean(numpy.var(middle_x, axis=0)),
                var_y = numpy.mean(numpy.var(middle_y, axis=0)),

                OutputLog().write('Layer - loss: {1}, correlation: {2}, recall: {3}, mean_x: {4}, mean_y: {5},'
                                  'var_x: {6}, var_y: {7}'.format(index,
                                                                  validation_loss,
                                                                  correlation,
                                                                  sum(search_recall) + sum(
                                                                      describe_recall),
                                                                  mean_x,
                                                                  mean_y,
                                                                  var_x,
                                                                  var_y))

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

            updates.update(
                lasagne.updates.nesterov_momentum(loss, params, learning_rate=current_learning_rate, momentum=0.9))
            del train_fn
            train_fn = theano.function([x_var, y_var], [loss] + outputs.values(), updates=updates)

    model_results['experiment'] = experiment_values

    with file(os.path.join(path, 'results_{0}.p'.format(id)), 'wb') as results_file:
        pickle.dump(model_results, results_file)

    Params.__dict__ = param_backup

    del train_fn
    del test_x
    del test_y
    del model_x
    del model_y

    return model_results


if __name__ == '__main__':

    data_set_config = sys.argv[1]
    if len(sys.argv) > 2:
        top = int(sys.argv[2])
    else:
        top = 0

    OutputLog().set_path(OUTPUT_DIR)
    OutputLog().set_verbosity('info')

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    Params.print_params()
    path = OutputLog().output_path

    process_pool = Pool(PROCESS_NUMBER)

    run_experiment_partial = functools.partial(run_experiment, data_parameters=data_parameters, path=path)

    with open(os.path.join(path, 'experiments.pkl'), 'wb') as experiment_file:
        pickle.dump(experiments, experiment_file)

    for index, experiment in enumerate(experiments):
        OutputLog().write('Starting Experiment {0}'.format(experiment))

        if isinstance(experiment, tuple) and isinstance(experiment[1], numpy.ndarray):
            mapping = [(experiment[0], float(value)) for value in experiment[1]]
        elif isinstance(experiment, tuple):
            mapping = [experiment]
        elif isinstance(experiment, list):
            mapping = experiment

        results = []
        if PROCESS_NUMBER > 1:
            results = process_pool.map(run_experiment_partial, mapping)
        else:
            for mapped in mapping:
                results.append(run_experiment_partial(mapped))