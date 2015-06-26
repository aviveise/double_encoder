import cv2
from MISC.logger import OutputLog
from MISC.utils import complete_rank

__author__ = 'aviv'

import sys

import numpy

from tester_base import TesterBase

from MISC.utils import calculate_mardia, calculate_reconstruction_error, match_error


class TraceCorrelationTester(TesterBase):
    def __init__(self, test_set_x, test_set_y, top=50):
        super(TraceCorrelationTester, self).__init__(test_set_x, test_set_y)

        self.top = top

    def _calculate_metric(self, x, y, transformer, print_row):

        tickFrequency = cv2.getTickFrequency()

        var_x = numpy.var(x, axis=0)
        var_y = numpy.var(y, axis=0)

        loss_var_x = numpy.mean(var_x)
        loss_var_y = numpy.mean(var_y)

        loss = calculate_reconstruction_error(x, y)
        matches = match_error(x, y)

        search_recall, describe_recall = complete_rank(x,y)

        start_tick = cv2.getTickCount()
        if self.top == 0:
            correlation = (calculate_mardia(x, y, 0) / x.shape[1]) * 100
        else:
            correlation = (calculate_mardia(x, y, self.top) / self.top) * 100

        current_time = cv2.getTickCount()
        OutputLog().write('calculated correlation, time: {0}'.format(((current_time - start_tick) / tickFrequency)),
                          'debug')

        print_row.append(correlation)
        print_row.append(loss)
        print_row.append(loss_var_x)
        print_row.append(loss_var_y)
        print_row.append(matches)
        print_row.append(sum(search_recall) + sum(describe_recall))

        return correlation

    def print_array(self, a):

        if a is None:
            return

        print ('array %d: ' % a.shape[0])

        for i in xrange(a.shape[0]):
            print '%f ' % a[i]

        print '\n'

    def _headers(self):

        return ['correlation', 'loss', '1/var_x', '1/var_y', 'match_error', 'recall']
