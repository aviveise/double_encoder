__author__ = 'aviv'

import sys

import numpy

from tester_base import TesterBase
from MISC.utils import calculate_mardia, calculate_trace, calculate_corrcoef


class TraceCorrelationTester(TesterBase):

    def __init__(self, test_set_x, test_set_y, top=50):
        super(TraceCorrelationTester, self).__init__(test_set_x, test_set_y)

        self.top = top

    def _find_correlation(self, x, y, transformer):

        x_var = numpy.var(x, axis=1)
        y_var = numpy.var(y, axis=1)

        x_mean = numpy.mean(x, axis=1)
        y_mean = numpy.mean(y, axis=1)

        print 'x variance: mean %f, var %f\n' % (numpy.mean(x_var), numpy.var(x_var))
        print 'y variance: mean %f, var %f\n' % (numpy.mean(y_var), numpy.var(y_var))

        print 'x mean: mean %f, var %f\n' % (numpy.mean(x_mean), numpy.var(x_mean))
        print 'y mean: mean %f, var %f\n' % (numpy.mean(y_mean), numpy.var(y_mean))
        sys.stdout.flush()

        print calculate_trace(x, y, self.top)
        sys.stdout.flush()

        print calculate_corrcoef(x, y, self.top)
        sys.stdout.flush()

        result = calculate_mardia(x, y, self.top)

        print result
        sys.stdout.flush()


    def print_array(self, a):

        if a is None:
            return

        print ('array %d: ' % a.shape[0])

        for i in xrange(a.shape[0]):
            print '%f ' % a[i]

        print '\n'
