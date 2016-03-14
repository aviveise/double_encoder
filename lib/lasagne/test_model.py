import ConfigParser
import os
import pickle
import sys

import lasagne
import numpy
from sklearn import preprocessing
from sklearn import cross_decomposition
from tabulate import tabulate
from theano import tensor, theano

from lib.MISC.logger import OutputLog
from lib.MISC.utils import ConfigSectionMap, calculate_reconstruction_error, complete_rank, calculate_mardia, center, \
    complete_rank_2
from lib.lasagne.Layers.TiedDropoutLayer import TiedDropoutLayer
from lib.lasagne.params import Params
from lib.MISC.container import Container



import lib.DataSetReaders

OUTPUT_DIR = r'C:\Workspace\output'
MODEL_PATH = r'C:\Workspace\output\2016_03_13_00_57_18'


def normalize(x, centeralize):
    x_c = x
    if centeralize:
        x_c = center(x)[0]

    x_n = preprocessing.normalize(x_c, axis=1)
    return x_n


def test_model(model_x, model_y, dataset_x, dataset_y, parallel=1, validate_all=True, top=0, cca=None):
    # Test
    y_values = model_x(dataset_x, dataset_y)
    x_values = model_y(dataset_x, dataset_y)

    if cca:
        x_values, y_values = cca.transform(x_values[Params.TEST_LAYER],
                                           y_values[Params.TEST_LAYER])

    OutputLog().write('\nTesting model\n')

    header = ['layer', 'loss', 'corr', 'search1', 'search5', 'search10', 'search_sum', 'desc1', 'desc5', 'desc10',
              'desc_sum']

    rows = []

    if validate_all:
        for index, (x, y) in enumerate(zip(x_values, y_values)):
            if data_set.x_y_mapping['test'] is not None:
                search_recall, describe_recall = complete_rank_2(x, y, data_set.x_y_mapping)
            else:
                search_recall, describe_recall = complete_rank(x, y, data_set.reduce_val)

            loss = calculate_reconstruction_error(x, y)
            correlation = calculate_mardia(x, y, top)

            print_row = ["{0}_raw".format(index), loss, correlation]
            print_row.extend(search_recall)
            print_row.append(sum(search_recall))
            print_row.extend(describe_recall)
            print_row.append(sum(describe_recall))

            rows.append(print_row)
    else:
        middle_x = x_values
        middle_y = y_values

        if data_set.x_y_mapping is None:
            search_recall, describe_recall = complete_rank_2(middle_x, middle_y, data_set.x_y_mapping)
        else:
            search_recall, describe_recall = complete_rank(middle_x, middle_y, data_set.reduce_val)

        loss = calculate_reconstruction_error(middle_x, middle_y)
        correlation = calculate_mardia(middle_x, middle_y, top)

        print_row = ["{0} ".format(Params.TEST_LAYER), loss, correlation]
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

    # cca = cross_decomposition.CCA(top)

    t_x = test_y(data_set.testset[0], data_set.testset[1])
    t_y = test_x(data_set.testset[0], data_set.testset[1])

    x_c = center(t_x)[0]
    y_c = center(t_y)[0]

    x_n = preprocessing.normalize(x_c, axis=1)
    y_n = preprocessing.normalize(y_c, axis=1)

    for i in range(0, 5 - x_n.shape[0] % 5):
        x_n = numpy.vstack((x_n, x_n[-1, :]))
        y_n = numpy.vstack((y_n, y_n[-1, :]))

    num_X_samples = x_n.shape[0]
    num_Y_samples = y_n.shape[0]

    y_x_mapping = None
    if y_x_mapping is None:
        x_n = x_n[0:x_n.shape[0]:5, :]
        y_x_mapping = numpy.repeat(numpy.arange(x_n.shape[0]), 5)
        num_X_samples = num_X_samples / 5

    y_x_sim_matrix = numpy.dot(x_n, y_n.T)

    recall_n_vals = [1, 5, 10]
    num_of_recall_n_vals = len(recall_n_vals)

    x_search_recall = numpy.zeros((num_of_recall_n_vals, 1))
    describe_x_recall = numpy.zeros((num_of_recall_n_vals, 1))

    x_search_sorted_neighbs = numpy.argsort(y_x_sim_matrix, axis=0)[::-1, :]
    x_search_ranks = numpy.array(
        [numpy.where(col == y_x_mapping[index])[0] for index, col in enumerate(x_search_sorted_neighbs.T)])

    for idx, recall in enumerate(recall_n_vals):
        x_search_recall[idx] = numpy.sum(x_search_ranks <= recall)

    x_search_recall = 100 * x_search_recall / num_Y_samples

    describe_y_sorted_neighbs = numpy.argsort(y_x_sim_matrix, axis=1)[:, ::-1]
    describe_y_ranks = numpy.array([numpy.where(numpy.in1d(row, numpy.where(y_x_mapping == index)[0]))[0]
                                    for index, row in enumerate(describe_y_sorted_neighbs)]).min(axis=1)

    for idx, recall in enumerate(recall_n_vals):
        describe_x_recall[idx] = numpy.sum(describe_y_ranks <= recall)

    describe_x_recall = 100 * describe_x_recall / num_X_samples

    print 'd_recall:{0}, s_recall:{0}'.format(describe_x_recall, x_search_recall)

    for i in range(20):
        print 'image {0} sentences {1}'.format(i, x_search_sorted_neighbs[i, 0:5])
        print 'sentence {0} images {1}'.format(i, describe_y_sorted_neighbs[i, 0:5])
    # cca.fit(train_x[Params.TEST_LAYER], train_y[Params.TEST_LAYER])

    # import pickle
    # cca = pickle.load(open(r'C:\Workspace\cca_mnist.p'))

    # test_model(test_x, test_y, data_set.testset[0], data_set.testset[1], parallel=5, top=top)
