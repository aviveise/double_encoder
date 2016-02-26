import ConfigParser
import os
import pickle
import sys

import h5py
import hickle
import numpy
from sklearn.decomposition import PCA
from theano import config

from lib.MISC.container import Container
from lib.MISC.logger import OutputLog
from lib.MISC.utils import ConfigSectionMap, normalize, scale_cols
import lib.DataSetReaders

OUTPUT_PATH = ''


if __name__ == '__main__':
    data_set_config = sys.argv[1]

    OutputLog().set_path(OUTPUT_PATH)
    OutputLog().set_verbosity('info')

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)

    data_set.preprocess()

    data_set.dump('_preprocessed')