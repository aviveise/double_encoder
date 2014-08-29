import os
import sys

import numpy
import numpy.linalg as linalg

from theano import config
__author__ = 'aviv'


class WhitenTransform(object):

    def __init__(self, data, maxD=-1):

        rowNum = data.T.shape[0]
        colNum = data.T.shape[1]

        alpha = 1 / float(colNum)
        ones = numpy.ones([colNum, 1], dtype=config.floatX)
        self._v = alpha * numpy.dot(data.T, ones)

        centered = data.T - numpy.dot(self._v, ones.T)

        k = min(colNum, rowNum)

        U, S, V = linalg.svd(centered, 0, 1)

        centered[:, 0: k] = U[:, 0: k]

        eps = numpy.exp(-8) * S[0]
        scale = numpy.sqrt(colNum - 1)
        n_new = 0

        for f in xrange(k):

            if S[f] >= eps:
                S[n_new] = S[f]
                centered[:, n_new] = centered[:, f] * (scale / S[f])
                n_new += 1

                if n_new == maxD:
                    break

        self._w = centered[:, 0: n_new].T

    def transform(self, data):

        numCol = data.T.shape[1]
        ones = numpy.ones([1, numCol], dtype=config.floatX)
        centered = data.T - numpy.dot(self._v, ones)
        transformedData = numpy.dot(self._w, centered)
        return transformedData.T