__author__ = 'aviv'

import sys

import numpy

from tester_base import TesterBase

from MISC.utils import calculate_mardia, calculate_trace, calculate_corrcoef, calculate_reconstruction_error


class TraceCorrelationTester(TesterBase):

    def __init__(self, test_set_x, test_set_y, top=50):
        super(TraceCorrelationTester, self).__init__(test_set_x, test_set_y)

        self.top = top

    def _calculate_metric(self, x, y, transformer, print_row):

        x_var = numpy.var(x, axis=0)
        y_var = numpy.var(y, axis=0)

        x_mean = numpy.mean(x, axis=0)
        y_mean = numpy.mean(y, axis=0)

        print_row.append(numpy.mean(x_var))
        print_row.append(numpy.max(x_var))
        print_row.append(numpy.mean(x_mean))
        print_row.append(numpy.max(x_mean))
        print_row.append(numpy.mean(y_var))
        print_row.append(numpy.max(y_var))
        print_row.append(numpy.mean(y_mean))
        print_row.append(numpy.max(y_mean))

        #svd_correlation = 0
        #correlation_coefficients = 0
        trace_correlation = 0

        try:

            trace_correlation = calculate_trace(x, y)
            #correlation_coefficients = calculate_corrcoef(x, y, self.top)
        except:
            print 'exception on loss calculation'

        loss = calculate_reconstruction_error(x, y)

        print_row.append(trace_correlation)
        print_row.append(loss)

        svd_correlation = calculate_mardia(x, y, self.top)
        print_row.append(svd_correlation)

        return trace_correlation

    def print_array(self, a):

        if a is None:
            return

        print ('array %d: ' % a.shape[0])

        for i in xrange(a.shape[0]):
            print '%f ' % a[i]

        print '\n'

    def _headers(self):

        return ['var_x (avg)',
                'var_x (max)',
                'mean_x (avg)',
                'mean_x (max)',
                'var_y (avg)',
                'var_y (max)',
                'mean_y (avg)',
                'mean_y (max)',
                'trace correlation',
                'correlation coef',
                'svd correlation',
                'loss']
