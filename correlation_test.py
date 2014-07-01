__author__ = 'aviv'

import os
import sys
import numpy

from MISC.utils import center, unitnorm
from numpy.linalg import svd

class CorrelationTest(object):

    def __init__(self, test_set_x, test_set_y):

        self._x = test_set_x
        self._y = test_set_y


    def test(self, tester):

        x_tilde, y_tilde = tester.compute_outputs(self._x, self._y)

        return self._find_correlation(x_tilde.T, y_tilde.T, True) / x_tilde.shape[1]


    def _find_correlation(self, x, y, svd_sum=True):

        forward = unitnorm(center(x))
        backward = unitnorm(center(y))

        if svd_sum:
            return svd(numpy.dot(forward, backward.T), compute_uv=False).sum()

        else:
            correlation = numpy.ndarray([forward.shape[0], 1])

            for i in xrange(forward.shape[0]):
                correlation[i] = numpy.corrcoef(forward, backward)[0, 1]

            return correlation.sum()


