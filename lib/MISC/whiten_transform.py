import os
import sys

import numpy
import numpy.linalg as linalg
import scipy

from theano import config, theano

__author__ = 'aviv'


class WhitenTransform(object):

    @staticmethod
    def fit(data, eps=0.01):

        centered = data - numpy.dot(numpy.mean(data, axis=0).reshape(data.shape[1], 1),
                                    numpy.ones((1, data.shape[0]), dtype=theano.config.floatX)).T
        sigma = numpy.dot(centered, centered.T) / data.shape[1]
        [U, S, V] = scipy.linalg.svd(sigma)
        return numpy.dot(numpy.dot(U, numpy.diag(1/numpy.sqrt(S + eps))), U.T)

        # rowNum = data.shape[0]
        # colNum = data.shape[1]
        #
        # alpha = 1 / float(colNum)
        # ones = numpy.ones([colNum, 1], dtype=config.floatX)
        # self._v = alpha * numpy.dot(data, ones)
        #
        # centered = data - numpy.dot(self._v, ones.T)
        #
        # k = min(colNum, rowNum)
        #
        # U, S, V = linalg.svd(centered, 0, 1)
        #
        # eps = numpy.exp(-8) * S[0]
        # scale = numpy.sqrt(colNum - 1)
        # n_new = 0
        #
        # self._w = numpy.ndarray([k, k], dtype=config.floatX)
        #
        # for f in xrange(k):
        #
        #     if S[f] >= eps:
        #         S[n_new] = S[f]
        #         self._w[n_new, :] = U[f, :] * (scale / S[f])
        #         n_new += 1
        #
        #         if n_new == maxD:
        #             break
        #
        # self._w = self._w[0:n_new, :]

    @staticmethod
    def transform(data, w):
        centered = data - numpy.dot(numpy.mean(data, axis=0).reshape(data.shape[1], 1),
                                    numpy.ones((1, data.shape[0]), dtype=theano.config.floatX)).T
        return numpy.dot(w, centered)