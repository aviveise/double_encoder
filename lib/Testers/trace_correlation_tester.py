__author__ = 'aviv'

import numpy

from tester_base import TesterBase
from MISC.utils import center, unitnorm

class TraceCorrelationTester(TesterBase):

    def __init__(self, test_set_x, test_set_y):
        super(TraceCorrelationTester, self).__init__(test_set_x, test_set_y)

    def _find_correlation(self, x, y, transformer):

        forward = unitnorm(center(x))
        backward = unitnorm(center(y))

        #cov = numpy.dot(forward, backward.T)
        #s = numpy.linalg.svd(cov,compute_uv=0)

        diag = numpy.abs(numpy.diagonal(numpy.dot(forward, backward.T)))
        diag.sort()
        diag = diag[::-1]

        return (numpy.sum(diag) / diag.shape[0]) * 100



