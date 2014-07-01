__author__ = 'aviv'

import sys
import io
import abc


class RegularizationBase(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, weight):
        self.weight = weight


    @abc.abstractmethod
    def compute(self, symmetric_double_encoder):
        return

