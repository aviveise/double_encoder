__author__ = 'aviv'

import numpy
from sklearn.cross_decomposition import CCA
from MISC.utils import center, unitnorm

class CorrelationTest(object):

    def __init__(self, test_set_x, test_set_y):

        self._x = test_set_x
        self._y = test_set_y

    def test(self, tester):

        x_tilde, y_tilde = tester.compute_outputs(self._x, self._y)

        return self._find_correlation(x_tilde.T, y_tilde.T, True)

    def _find_correlation(self, x, y, svd_sum=True):

        forward = unitnorm(center(x))
        backward = unitnorm(center(y))

        cov = numpy.dot(forward, backward.T)
        s = numpy.linalg.svd(cov,compute_uv=0)

        print cov.sum()
        print s.sum()

        diag = numpy.abs(numpy.diagonal(numpy.dot(forward, backward.T)))
        diag.sort()
        diag = diag[::-1]

        print numpy.sum(diag)

        return numpy.sum(diag) / diag.shape[0]

        #if svd_sum:
        #    return svd(numpy.dot(forward, backward.T), compute_uv=False).sum()

        #else:
        #    correlation = numpy.ndarray([forward.shape[0], 1])

        #   for i in xrange(forward.shape[0]):
        #        correlation[i] = numpy.corrcoef(forward, backward)[0, 1]

        #    return correlation.sum()


