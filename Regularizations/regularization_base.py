__author__ = 'aviv'

import sys
import io
import abc

from MISC.logger import OutputLog


class RegularizationBase(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, regularization_parameters):

        OutputLog().write('Adding regularization: ' + regularization_parameters['type'])

        self.weight = regularization_parameters['weight']


    @abc.abstractmethod
    def compute(self, symmetric_double_encoder):
        return

