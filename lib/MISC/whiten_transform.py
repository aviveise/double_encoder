import os
import sys

import numpy
import numpy.linalg as linalg

from theano import config
__author__ = 'aviv'


class WhitenTransform(object):

    def __init__(self, data, maxD=-1):

        rowNum = data.shape[0]
        colNum = data.shape[1]

        alpha = 1 / float(colNum)
        ones = numpy.ones([colNum, 1], dtype=config.floatX)
        self._v = alpha * numpy.dot(data, ones)

        centered = data - numpy.dot(self._v, ones.T)

        k = min(colNum, rowNum)

        U, S, V = linalg.svd(centered, 0, 1)

        eps = numpy.exp(-8) * S[0]
        scale = numpy.sqrt(colNum - 1)
        n_new = 0

        self._w = numpy.ndarray([k, k], dtype=config.floatX)

        for f in xrange(k):

            if S[f] >= eps:
                S[n_new] = S[f]
                self._w[n_new, :] = U[f, :] * (scale / S[f])
                n_new += 1

                if n_new == maxD:
                    break

        self._w = self._w[0:n_new, :]

    def transform(self, data):

        numCol = data.shape[1]
        ones = numpy.ones([1, numCol], dtype=config.floatX)
        centered = data - numpy.dot(self._v, ones)
        transformedData = numpy.dot(self._w, centered)
        return transformedData