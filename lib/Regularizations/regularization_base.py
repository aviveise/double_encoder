__author__ = 'aviv'

import abc

from lib.MISC.logger import OutputLog


class RegularizationBase(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, regularization_parameters):

        OutputLog().write('Adding regularization: ' + regularization_parameters['type'])

        self.weight = float(regularization_parameters['weight'])
        self.regularization_type = regularization_parameters['type']

    @abc.abstractmethod
    def compute(self, symmetric_double_encoder):
        return


    def print_regularization(self, output_stream):

        output_stream.write('regularization_type: %s' % self.regularization_type)
        output_stream.write('regularization_weight: %f' % self.weight)