__author__ = 'aviv'

import numpy

from tester_base import TesterBase
from MISC.utils import center, unitnorm

class TraceCorrelationTester(TesterBase):

    def __init__(self, test_set_x, test_set_y, top=50):
        super(TraceCorrelationTester, self).__init__(test_set_x, test_set_y)

        self.top = top

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


        x_var = numpy.var(x, axis=1)
        y_var = numpy.var(y, axis=1)

        x_mean = numpy.mean(x, axis=1)
        y_mean = numpy.mean(y, axis=1)

        print 'x variance: mean %f, var %f\n' % (numpy.mean(x_var), numpy.var(x_var))
        print 'y variance: mean %f, var %f\n' % (numpy.mean(y_var), numpy.var(y_var))

        print 'x mean: mean %f, var %f\n' % (numpy.mean(x_mean), numpy.var(x_mean))
        print 'y mean: mean %f, var %f\n' % (numpy.mean(y_mean), numpy.var(y_mean))

        forward = unitnorm(center(x))
        backward = unitnorm(center(y))

        corr = numpy.dot(forward, backward.T)

        diag = numpy.abs(numpy.diagonal(corr))

        print 'correlations: %f\n' % numpy.sum(diag)
        print 'cross correlations: %f\n' % (numpy.sum(numpy.abs(corr)) - numpy.sum(diag))

        diag.sort()
        diag = diag[::-1]

        print 'correlations:\n'
        print diag

        return sum(diag[0:self.top + 1])


    def print_array(self, a):

        if a is None:
            return

        print ('array %d: ' % a.shape[0])

        for i in xrange(a.shape[0]):
            print '%f ' % a[i]

        print '\n'
