import ConfigParser
import os
import sys

import h5py
import hickle
import numpy
from theano import config

from lib.MISC.container import Container
from lib.MISC.logger import OutputLog
from lib.MISC.utils import ConfigSectionMap, normalize
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

    CNN_mat = h5py.File(os.path.join(data_set.dataset_path, 'visual_dataset_info.mat'), 'r')
    idx_mat = h5py.File(os.path.join(data_set.dataset_path, 'dataset_info_idx.mat'), 'r')

    training_sen_idx = idx_mat['trn_sent_I']
    training_image_idx = idx_mat['trn_images_I']

    validation_sen_idx = idx_mat['dev_sent_I']
    validation_image_idx = idx_mat['dev_images_I']

    test_sen_idx = idx_mat['tst_sent_I']
    test_image_idx = idx_mat['tst_images_I']

    if data_set._full:
        train_size = training_sen_idx.shape[0]
        data_set.reduce_val = 5
    else:
        train_size = training_image_idx.shape[0]
        data_set.reduce_val = 0

    dev_size = validation_sen_idx.shape[0]
    test_size = test_sen_idx.shape[0]

    print 'Loading Images'
    CNN_output = CNN_mat['image_vecs']
    images_sent_mapping = CNN_mat['image_idx_of_sent'].value

    trainset_x = numpy.ndarray((train_size, CNN_output.shape[1]), dtype=config.floatX)
    tuning_x = numpy.ndarray((dev_size, CNN_output.shape[1]), dtype=config.floatX)
    testset_x = numpy.ndarray((test_size, CNN_output.shape[1]), dtype=config.floatX)

    for i in range(train_size):
        trainset_x[i, :] = CNN_output[int(images_sent_mapping[int(training_sen_idx[i]) - 1]) - 1]

    for i in range(dev_size):
        tuning_x[i, :] = CNN_output[int(images_sent_mapping[int(validation_sen_idx[i]) - 1]) - 1]

    for i in range(test_size):
        testset_x[i, :] = CNN_output[int(images_sent_mapping[int(test_sen_idx[i]) - 1]) - 1]

    hickle.dump({'train': trainset_x, 'tune':tuning_x, 'test':testset_x}, open(os.path.join(OUTPUT_PATH, 'x.p')))

    del trainset_x
    del testset_x
    del tuning_x
    del CNN_output
    del CNN_mat
    del images_sent_mapping

    FV_mat = h5py.File(os.path.join(data_set.dataset_path, 'sentences_FV.mat'), 'r')
    feature_vectors = FV_mat['sent_vecs']

    trainset_y = numpy.ndarray((train_size, feature_vectors[0].shape[0]), dtype=config.floatX)
    tuning_y = numpy.ndarray((dev_size, feature_vectors[0].shape[0]), dtype=config.floatX)
    testset_y = numpy.ndarray((testset_x, feature_vectors[0].shape[0]), dtype=config.floatX)

    for i in range(train_size):
        trainset_y[i, :] = feature_vectors[int(training_sen_idx[i]) - 1]

    for i in range(dev_size):
        tuning_y[i, :] = feature_vectors[int(validation_sen_idx[i]) - 1]

    for i in range(test_size):
        testset_x[i, :] = feature_vectors[int(test_sen_idx[i]) - 1]

    hickle.dump({'train': trainset_y, 'tune':tuning_y, 'test':testset_y}, open(os.path.join(OUTPUT_PATH, 'y.p')))

    del trainset_y
    del testset_y
    del tuning_y
    del feature_vectors
    del FV_mat
