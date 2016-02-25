import ConfigParser
import os
import pickle
import sys

import lasagne
from sklearn import preprocessing
from tabulate import tabulate
from theano import tensor, theano

from lib.MISC.logger import OutputLog
from lib.MISC.utils import ConfigSectionMap, calculate_reconstruction_error, complete_rank, calculate_mardia, center
from lib.lasagne.Layers.TiedDropoutLayer import TiedDropoutLayer
from lib.lasagne.params import Params
from lib.MISC.container import Container

import lib.DataSetReaders

OUTPUT_DIR = r'C:\Workspace\output'
MODEL_PATH = r'C:\Workspace\output\2016_02_12_20_34_50'


def normalize(x, centeralize):
    x_c = x
    if centeralize:
        x_c = center(x)[0]

    x_n = preprocessing.normalize(x_c, axis=1)
    return x_n


def test_model(model_x, model_y, dataset_x, dataset_y, parallel=1, validate_all=True, top=0):
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
            correlation = calculate_mardia(x, y, top)

            print_row = ["{0}_raw".format(index), loss, correlation]
            print_row.extend(search_recall)
            print_row.append(sum(search_recall))
            print_row.extend(describe_recall)
            print_row.append(sum(describe_recall))

            rows.append(print_row)

    OutputLog().write(tabulate(rows, headers=header))


if __name__ == '__main__':

    data_set_config = sys.argv[1]
    if len(sys.argv) > 2:
        top = int(sys.argv[2])
    else:
        top = 0

    model_x = pickle.load(open(os.path.join(MODEL_PATH, 'model_x.p')))
    model_y = pickle.load(open(os.path.join(MODEL_PATH, 'model_y.p')))

    OutputLog().set_path(OUTPUT_DIR)
    OutputLog().set_verbosity('info')

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)
    data_set.load()

    x_var = model_x[0].input_var
    y_var = model_y[0].input_var


    # Export network
    path = OutputLog().output_path

    hidden_x = filter(lambda layer: isinstance(layer, TiedDropoutLayer), model_x)
    hidden_y = filter(lambda layer: isinstance(layer, TiedDropoutLayer), model_y)
    hidden_y = reversed(hidden_y)

    test_y = theano.function([x_var, y_var],
                             [lasagne.layers.get_output(layer, deterministic=True) for layer in
                              hidden_x],
                             on_unused_input='ignore')
    test_x = theano.function([x_var, y_var],
                             [lasagne.layers.get_output(layer, deterministic=True) for layer in
                              hidden_y],
                             on_unused_input='ignore')

    batch_number = data_set.trainset[0].shape[0] / Params.BATCH_SIZE

    OutputLog().write('Test results')

    test_model(test_x, test_y, data_set.testset[0], data_set.testset[1], parallel=5, top=top)