import ConfigParser
import sys
from collections import OrderedDict
import lasagne
import numpy
from theano import tensor, theano
from lib.MISC.container import Container
from lib.MISC.logger import OutputLog
from lib.MISC.utils import ConfigSectionMap, complete_rank, calculate_reconstruction_error, calculate_mardia
from lib.lasagne.Models import parallel_model, bisimilar_model, iterative_model, tied_dropout_iterative_model
from lib.lasagne.learnedactivations import batchnormalizeupdates
import lib.DataSetReaders

OUTPUT_DIR = r'C:\Workspace\output'
BATCH_SIZE = 128
EPOCH_NUMBER = 1


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


if __name__ == '__main__':

    data_set_config = sys.argv[1]

    OutputLog().set_path(OUTPUT_DIR)
    OutputLog().set_verbosity('info')

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)

    var_y = tensor.fmatrix()
    var_x = tensor.fmatrix()

    model = iterative_model

    OutputLog().write('Model: {0}'.format(model.__name__))

    layer_sizes = [2048, 2048, 2048]
    drop_prob = [0.5, 0.5, 0.5]

    model_x, hidden_x, weights_x, biases_x, prediction_y, hooks_x = model.build_single_channel(var_x,
                                                                                               data_set.trainset[
                                                                                                   0].shape[1],
                                                                                               data_set.trainset[
                                                                                                   1].shape[1],
                                                                                               layer_sizes=layer_sizes,
                                                                                               drop_prob=drop_prob,
                                                                                               name='x')

    loss_y = lasagne.objectives.squared_error(var_y, prediction_y).sum(axis=1).mean()

    params_x = lasagne.layers.get_all_params(model_x, trainable=True)
    updates = OrderedDict(batchnormalizeupdates(hooks_x, 100))
    updates.update(lasagne.updates.nesterov_momentum(loss_y, params_x, 0.001, 0.5))

    train_fn_x = theano.function([var_x, var_y], [loss_y], updates=updates)

    batch_number = data_set.trainset[0].shape[0] / BATCH_SIZE

    for epoch in range(EPOCH_NUMBER):
        OutputLog().write('Epoch {0}'.format(epoch))
        for index, batch in enumerate(
                iterate_minibatches(data_set.trainset[0], data_set.trainset[1], BATCH_SIZE, True)):
            input_x, input_y = batch
            loss = train_fn_x(input_x, input_y)
            OutputLog().write('{0}/{1} loss: {2}'.format(index, batch_number, loss[0]))

    model_y, hidden_y, weights_y, biases_y, prediction_x, hooks_y = model.build_single_channel(var_y,
                                                                                               data_set.trainset[
                                                                                                   1].shape[1],
                                                                                               data_set.trainset[
                                                                                                   0].shape[1],
                                                                                               layer_sizes=layer_sizes,
                                                                                               weight_init=[
                                                                                                   W.get_value().T for W
                                                                                                   in
                                                                                                   reversed(weights_x)],
                                                                                               bias_init=list(reversed(
                                                                                                   biases_x))[1:] + [
                                                                                                             lasagne.init.Constant(
                                                                                                                 0.)],
                                                                                               drop_prob=drop_prob,
                                                                                               name='y')

    loss_x = lasagne.objectives.squared_error(var_x, prediction_x).sum(axis=1).mean()

    params_y = lasagne.layers.get_all_params(model_y, trainable=True)
    updates = OrderedDict(batchnormalizeupdates(hooks_y, 100))
    updates.update(lasagne.updates.nesterov_momentum(loss_x, params_y, 0.001, 0.5))

    train_fn_y = theano.function([var_x, var_y], [loss_x], updates=updates)

    test_y = theano.function([var_x, var_y],
                             [lasagne.layers.get_output(layer, moving_avg_hooks=hooks_x, deterministic=True) for layer
                              in
                              hidden_x],
                             on_unused_input='ignore')
    test_x = theano.function([var_x, var_y],
                             [lasagne.layers.get_output(layer, moving_avg_hooks=hooks_y, deterministic=True) for layer
                              in
                              reversed(hidden_y)],
                             on_unused_input='ignore')

    for epoch in range(EPOCH_NUMBER):
        OutputLog().write('Epoch {0}'.format(epoch))
        for index, batch in enumerate(
                iterate_minibatches(data_set.trainset[0], data_set.trainset[1], BATCH_SIZE, True)):
            input_x, input_y = batch
            loss = train_fn_y(input_x, input_y)
            OutputLog().write('{0}/{1} loss: {2}'.format(index, batch_number, loss[0]))

        x_values = test_y(data_set.tuning[0], data_set.tuning[1])
        y_values = test_x(data_set.tuning[0], data_set.tuning[1])

        OutputLog().write('Validating model')

        for index, (x, y) in enumerate(zip(x_values, y_values)):
            search_recall, describe_recall = complete_rank(x, y, data_set.reduce_val)
            loss = calculate_reconstruction_error(x, y)
            correlation = calculate_mardia(x, y, 0)

            OutputLog().write('Layer {0} - loss: {1}, correlation: {2}, recall: {3}'.format(index,
                                                                                            loss,
                                                                                            correlation,
                                                                                            sum(search_recall) + sum(
                                                                                                describe_recall)))
