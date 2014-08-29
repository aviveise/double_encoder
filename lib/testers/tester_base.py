__author__ = 'aviv'

import sys
import os
import abc

class TesterBase(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, correlation_optimizer):
        self._correlation_optimizer = correlation_optimizer
        return

    @abc.abstractmethod
    def compute_outputs(self, test_set_x, test_set_y):
        return


