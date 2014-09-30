__author__ = 'aviv'

import numpy
from sklearn.cross_decomposition import CCA
from MISC.utils import center, unitnorm
from tester_base import TesterBase


class TraceCorrelationTester(TesterBase):

    def __init__(self, test_set_x, test_set_y):
        super(TraceCorrelationTester, self).__init__(test_set_x, test_set_y)

    def _find_correlation(self, x, y, transformer):

        forward = unitnorm(center(x))
        backward = unitnorm(center(y))

        cov = numpy.dot(forward, backward.T)
        s = numpy.linalg.svd(cov,compute_uv=0)

        p = numpy.corrcoef(x, y)[0:x.shape[0] - 1, x.shape[0]:(2 * x.shape[0] - 1)]

        print cov.sum()
        print s.sum()
        print p.sum()
        print p.diagonal().sum()


        diag = numpy.abs(numpy.diagonal(numpy.dot(forward, backward.T)))
        diag.sort()
        diag = diag[::-1]

        print numpy.sum(diag)

        return (numpy.sum(diag) / diag.shape[0]) * 100



