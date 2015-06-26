__author__ = 'aviv'

import sys

import numpy

from tester_base import TesterBase

from MISC.utils import calculate_mardia, calculate_trace, calculate_corrcoef


class TraceCorrelationTester(TesterBase):

    def __init__(self, test_set_x, test_set_y, top=50):
        super(TraceCorrelationTester, self).__init__(test_set_x, test_set_y)

        self.top = top

    def _calculate_metric(self, x, y, transformer):



