import ConfigParser
import os
import sys

import cPickle

import numpy

from lib.MISC.container import Container
from lib.MISC.logger import OutputLog
from lib.MISC.utils import ConfigSectionMap

import lib.DataSetReaders


def dump(dataset_x, dataset_y, type, path):
    numpy.save(os.path.join(path, type + '_x'), dataset_x)
    numpy.save(os.path.join(path, type + '_y'), dataset_y)

OUTPUT_DIR = r'/home/avive/workspace/output'

if __name__ == '__main__':

    data_set_config = sys.argv[1]

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    OutputLog().set_path(OUTPUT_DIR)
    OutputLog().set_verbosity('info')

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)
    data_set.load()

    path = data_set.dataset_path
    if not os.path.isdir(path):
        path = os.path.dirname(os.path.abspath(path))

    try:
        dump(data_set.trainset[0], data_set.trainset[1], 'train', path)
        dump(data_set.testset[0], data_set.testset[1], 'test', path)
        dump(data_set.tuning[0], data_set.tuning[1], 'validate', path)

        numpy.save(os.path.join(path, 'mapping_train'), data_set.x_y_mapping['train'])
        numpy.save(os.path.join(path, 'mapping_test'), data_set.x_y_mapping['test'])
        numpy.save(os.path.join(path, 'mapping_dev'), data_set.x_y_mapping['dev'])

        with open(os.path.join(path, 'reduce.p'), 'w') as reduce_file:
            cPickle.dump(data_set.x_reduce, reduce_file)

        with open(os.path.join(path, 'params.p'), 'wb') as params_file:
            cPickle.dump(data_set.data_set_parameters, params_file, cPickle.HIGHEST_PROTOCOL)

    except Exception as e:
        OutputLog().write('Failed converting dataset with exception {}'.format(e))
