import os
import sys

import numpy
import cPickle
import gzip

if __name__ == '__main__':

    dataset_path = './mnist.pkl.gz'

    mnist_file = gzip.open(dataset_path, 'rb')
    train_set, valid_set, test_set = cPickle.load(mnist_file)

    train_set = (numpy.concatenate([train_set[0], valid_set[0]]), numpy.concatenate([train_set[1], valid_set[1]]))
    train_set_first = numpy.ndarray((60000, 392))
    train_set_second = numpy.ndarray((60000, 392))

    test_set_first = numpy.ndarray((10000, 392))
    test_set_second = numpy.ndarray((10000, 392))

    for index in xrange(test_set_first.shape[0]):
        image = train_set[0][index, :].reshape((28, 28)).reshape((784, 1), order='F')
        image_first = image[0:392]
        image_second = image[392:784]
        train_set_first[index, :] = image_first[:, 0]
        train_set_second[index, :] = image_second[:, 0]

    for index in xrange(test_set_second.shape[0]):
        image = test_set[0][index, :].reshape((28, 28)).reshape((784, 1), order='F')
        image_first = image[0:392]
        image_second = image[392:784]
        test_set_first[index, :] = image_first[:, 0]
        test_set_second[index, :] = image_second[:, 0]

    training_set = train_set_first, train_set_second
    test_set = test_set_first, test_set_second

    f = gzip.open('./mnist.split.gzip.pkl', mode='w+')

    cPickle.dump((training_set, test_set), f)

    f.close()