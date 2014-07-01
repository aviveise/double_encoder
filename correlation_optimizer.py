__author__ = 'aviv'

import os
import sys
import abc


class CorrelationOptimizer(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, training_set_x, training_set_y):

        self._x = training_set_x
        self._y = training_set_y

    @abc.abstractmethod
    def reconstruct_x(self):
        return

    @abc.abstractmethod
    def reconstruct_y(self):
        return
