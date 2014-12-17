__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor

class WeightDecayRegularization(RegularizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(WeightDecayRegularization, self).__init__(regularization_parameters)
        self._zeroing_param = float(regularization_parameters['zeroing_param'])

    def compute(self, symmetric_double_encoder):

        regularization = self._compute_L2(symmetric_double_encoder)
        regularization = regularization - self._zeroing_param

        return (self.weight / 2) * regularization * (regularization > 0)

    def _compute_L2(self, symmetric_double_encoder):

        regularization = 0

        for layer in symmetric_double_encoder:
            regularization += Tensor.sum(layer.Wx ** 2)
            regularization += Tensor.sum(layer.Wy ** 2)

        return regularization

    def print_regularization(self, output_stream):
        super(WeightDecayRegularization, self).print_regularization(output_stream)
