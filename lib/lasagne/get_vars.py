import ConfigParser
import os
import sys

import cPickle

import lasagne
import numpy
import theano

from collections import OrderedDict
from lib.MISC.container import Container
from lib.MISC.logger import OutputLog
from lib.MISC.utils import ConfigSectionMap
from lib.lasagne.Layers.TiedDropoutLayer import TiedDropoutLayer
from lib.lasagne.learnedactivations import BatchNormalizationLayer
from lib.lasagne.params import Params
from lasagne.layers.noise import DropoutLayer
from math import ceil, floor

import lib.DataSetReaders

OUTPUT_DIR = r'/media/data1/aviveise/results/'
INPUT_DIR = r'/media/data1/aviveise/results/flickr8k'
MEMORY_LIMIT = 8000000.

def iterate_single_minibatch(inputs, batchsize, shuffle=False, preprocessor=None):
    if shuffle:
        indices = numpy.arange(len(inputs))
        numpy.random.shuffle(indices)

    batch_limit = min(len(inputs), ceil(MEMORY_LIMIT / inputs.shape[1] / batchsize / 4.))

    buffer = numpy.load(inputs.filename, 'r')

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if (start_idx / batchsize) % batch_limit == 0:
            buffer = numpy.load(inputs.filename, 'r')

        if preprocessor is not None:
            yield preprocessor(numpy.copy(buffer[excerpt]))
        else:
            yield buffer[excerpt]


def get_vars(model_x, model_y, data_set):

    x_total_value = None
    y_total_value = None

    x_var = model_x[0].input_var
    y_var = model_y[0].input_var

    hidden_x = filter(lambda layer: isinstance(layer, TiedDropoutLayer), model_x)
    hidden_y = filter(lambda layer: isinstance(layer, TiedDropoutLayer), model_y)

    if len(hidden_x) == 0:
	hidden_x = filter(lambda layer: isinstance(layer, DropoutLayer), model_x)
    	hidden_y = filter(lambda layer: isinstance(layer, DropoutLayer), model_y)
    if len(hidden_x) == 0:
	hidden_x = filter(lambda layer: isinstance(layer, BatchNormalizationLayer), model_x)
        hidden_y = filter(lambda layer: isinstance(layer, BatchNormalizationLayer), model_y)


    hidden_y = reversed(hidden_y)

    hooks = OrderedDict()

    test_y = theano.function([x_var],
                            [lasagne.layers.get_output(layer, moving_avg_hooks=hooks, deterministic=True) for layer in hidden_x],
                            on_unused_input='ignore')

    test_x = theano.function([y_var],
                             [lasagne.layers.get_output(layer, moving_avg_hooks=hooks, deterministic=True) for layer in hidden_y],
                             on_unused_input='ignore')

    tuning_x = data_set.tuning[0]
    tuning_y = data_set.tuning[1]

    for index, batch in enumerate(
            iterate_single_minibatch(tuning_x, Params.VALIDATION_BATCH_SIZE, False, preprocessor=data_set.preprocessors[0])):
        x_values = test_y(batch)[Params.TEST_LAYER]

        if x_total_value is None:
            x_total_value = x_values
        else:
            x_total_value = numpy.vstack((x_total_value, x_values))

    for index, batch in enumerate(
            iterate_single_minibatch(tuning_y, Params.VALIDATION_BATCH_SIZE, False, preprocessor=data_set.preprocessors[1])):

        y_values = test_x(batch)[Params.TEST_LAYER]

        if y_total_value is None:
            y_total_value = y_values
        else:
            y_total_value = numpy.vstack((y_total_value, y_values))

    print 'var_x: {0}, var_y: {1}'.format(numpy.mean(numpy.var(x_total_value, axis=0)),
                                          numpy.mean(numpy.var(y_total_value, axis=0)))


if __name__ == '__main__':

    data_set_config = sys.argv[1]

    results_folder = os.path.join(OUTPUT_DIR, str.rstrip(str.split(data_set_config, '_')[-1][:-4]))

    OutputLog().set_path(results_folder)
    OutputLog().set_verbosity('info')

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)
    data_set.load()

    dirs = []

    for (dirpath, dirnames, filenames) in os.walk(INPUT_DIR):
        dirs.extend(dirnames)

    for dir in dirs:
        print dir
        model_x_path = os.path.join(INPUT_DIR, dir, 'model_x.p')
	model_y_path = os.path.join(INPUT_DIR, dir, 'model_y.p')
	if not os.path.exists(model_x_path) or not os.path.exists(model_y_path):
		continue
        print '{0} Vars:'.format(dir)
        model_x = cPickle.load(open(model_x_path, 'rb'))
        model_y = cPickle.load(open(model_y_path, 'rb'))
        get_vars(model_x, model_y, data_set)

