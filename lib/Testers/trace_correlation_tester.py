import cv2
from MISC.logger import OutputLog

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

        #x_var = numpy.var(x)
        #y_var = numpy.var(y)

        #x_mean = numpy.mean(x)
        #y_mean = numpy.mean(y)

        h_x = numpy.dot(x, x.T)
        h_y = numpy.dot(y, y.T)

        x_var = numpy.var(h_x)
        y_var = numpy.var(h_y)

        x_mean = x_var
        y_mean = y_var

        print_row.append(x_var)
        print_row.append(x_var)
        print_row.append(x_mean)
        print_row.append(x_mean)
        print_row.append(y_var)
        print_row.append(y_var)
        print_row.append(y_mean)
        print_row.append(y_mean)

        start_tick = cv2.getTickCount()
        tickFrequency = cv2.getTickFrequency()

        trace_correlation = 0 #calculate_trace(x, y)

        current_time = cv2.getTickCount()
        OutputLog().write('calculated trace, time: {0}'.format(((current_time - start_tick) / tickFrequency)))

        start_tick = cv2.getTickCount()
        loss = 0 #calculate_reconstruction_error(x, y)

        current_time = cv2.getTickCount()
        OutputLog().write('calculated loss, time: {0}'.format(((current_time - start_tick) / tickFrequency)))

        #start_tick = cv2.getTickCount()
        #svd_correlation = calculate_mardia(x, y, self.top)

        #current_time = cv2.getTickCount()
        #OutputLog().write('calculated svd, time: {0}'.format(((current_time - start_tick) / tickFrequency)))

        print_row.append(trace_correlation)
        #print_row.append(svd_correlation)
        print_row.append(loss)

        return trace_correlation, min(x_var, y_var)

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
                #'svd correlation',
                'loss']
