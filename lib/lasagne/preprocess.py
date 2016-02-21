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
from lib.MISC.utils import ConfigSectionMap, normalize, scale_cols
import lib.DataSetReaders

OUTPUT_PATH = ''


if __name__ == '__main__':
    data_set_config = sys.argv[1]

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)

    with open('x.p', 'r') as x_file:
        x = hickle.load(x_file)

    trainset_x = x['train']
    tuning_x = x['tune']
    testset_x = x['testset']

    suffix = ''

    if data_set.normalize_data[0]:
        trainset_x, normalizer_x = normalize(trainset_x)
        tuning_x = normalizer_x.transform(tuning_x)
        testset_x = normalizer_x.transform(testset_x)
        scaler_x_path = os.path.join(OUTPUT_PATH, 'scaler_x.p')

        pickle.dump(normalizer_x, file(scaler_x_path, 'w'))

        suffix += '_normalized'

    if data_set.scale[0]:
        trainset_x, scaler_x = scale_cols(trainset_x)
        tuning_x = scaler_x.transform(tuning_x)
        testset_x = scaler_x.transform(testset_x)
        scaler_x_path = os.path.join(OUTPUT_PATH, 'scaler_x.p')

        pickle.dump(scaler_x, file(scaler_x_path, 'w'))

        suffix += '_scaled'

    if data_set.scale_rows:
        trainset_x = scale_cols(trainset_x.T)[0].T
        tuning_x = scale_cols(tuning_x.T)[0].T
        testset_x = scale_cols(testset_x.T)[0].T

        suffix += '_rows_scaled'

    if not data_set.pca[0] == 0:
        pca_dim1 = PCA(data_set.pca[0], data_set.whiten)
        pca_dim1.fit(trainset_x)

        trainset_x = pca_dim1.transform(trainset_x)
        tuning_x = pca_dim1.transform(tuning_x)
        testset_x = pca_dim1.transform(testset_x)

        suffix += '_pca{0}'.format(data_set.pca[0])

    with open('x_{0}.p'.format(suffix), 'w') as x_file:
        x = hickle.dump({'train':trainset_x,
                         'tune':tuning_x,
                         'testset':testset_x}, x_file)

    del trainset_x
    del tuning_x
    del testset_x

    with open('y.p', 'r') as y_file:
        y = hickle.load(y_file)

    trainset_y = y['train']
    tuning_y = y['tune']
    testset_y = y['testset']

    suffix = ''

    if data_set.normalize_data[1]:
        trainset_y, normalizer_y = normalize(trainset_y)
        tuning_y = normalizer_y.transform(tuning_y)
        testset_y = normalizer_y.transform(testset_y)
        scaler_y_path = os.path.join(OUTPUT_PATH, 'scaler_y.p')

        pickle.dump(normalizer_y, file(scaler_y_path, 'w'))

        suffix += '_normalized'

    if data_set.scale[1]:
        trainset_y, scaler_y = scale_cols(trainset_y)
        tuning_y = scaler_y.transform(tuning_y)
        testset_y = scaler_y.transform(testset_y)
        scaler_y_path = os.path.join(OUTPUT_PATH, 'scaler_y.p')

        pickle.dump(scaler_y, file(scaler_y_path, 'w'))

        suffix += '_scaled'

    if data_set.scale_rows:
        trainset_y = scale_cols(trainset_y.T)[0].T
        tuning_y = scale_cols(tuning_y.T)[0].T
        testset_y = scale_cols(testset_y.T)[0].T

        suffix += '_rows_scaled'

    if not data_set.pca[1] == 0:
        pca_dim1 = PCA(data_set.pca[1], data_set.whiten)
        pca_dim1.fit(trainset_y)

        trainset_y = pca_dim1.transform(trainset_y)
        tuning_y = pca_dim1.transform(tuning_y)
        testset_y = pca_dim1.transform(testset_y)

        suffix += '_pca{0}'.format(data_set.pca[1])

    with open('y_{0}.p'.format(suffix), 'w') as y_file:
        x = hickle.dump({'train':trainset_y,
                         'tune':tuning_y,
                         'testset':testset_y}, y_file)

    del trainset_y
    del tuning_y
    del testset_y