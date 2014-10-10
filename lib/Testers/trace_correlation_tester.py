__author__ = 'aviv'

import numpy

from tester_base import TesterBase
from MISC.utils import center, unitnorm

class TraceCorrelationTester(TesterBase):

    def __init__(self, test_set_x, test_set_y):
        super(TraceCorrelationTester, self).__init__(test_set_x, test_set_y)

    def _find_correlation(self, x, y, transformer):

        #u_x, s_x, v_x = numpy.linalg.svd(forward)
        #u_y, s_y, v_y = numpy.linalg.svd(backward)

        #k = min(forward.shape[1], forward.shape[0])

        #temp_x = numpy.dot(u_x[:, 0:k], v_x[0:k, :])
        #temp_y = numpy.dot(u_y[:, 0:k], v_y[0:k, :])

        #corr = numpy.dot(temp_x, temp_y.T)

        #s = numpy.linalg.svd(corr, compute_uv=False)

        #cov = numpy.dot(forward, backward.T)
        #s = numpy.linalg.svd(cov,compute_uv=0)

        #return (numpy.sum(s) / s.shape[0]) * 100

        #forward = unitnorm(center(x))
        #backward = unitnorm(center(y))

        forward = unitnorm(center(x))
        backward = unitnorm(center(y))

        diag = numpy.abs(numpy.diagonal(numpy.dot(forward, backward.T)))
        diag.sort()
        diag = diag[::-1]

        print sum(diag[0:50])

        return sum(diag[0:50])


