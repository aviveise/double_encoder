__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor


class HingeRegularization(RegularizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(HingeRegularization, self).__init__(regularization_parameters)

        self._k = float(regularization_parameters['k'])

    def compute(self, symmetric_double_encoder, params):

        regularization = 0
        for layer in symmetric_double_encoder:
            hidden_x = layer.output_forward_x
            hidden_y = layer.output_forward_y

            hinge_x = 1 - Tensor.abs_(hidden_x / self._k)
            hinge_y = 1 - Tensor.abs_(hidden_y / self._k)

            regularization += Tensor.sum(hinge_x * (hinge_x > 0))
            regularization += Tensor.sum(hinge_y * (hinge_y > 0))

        regularization -= self._zeroing_param

        return self.weight * regularization * (regularization > 0)

    def print_regularization(self, output_stream):
        super(HingeRegularization, self).print_regularization(output_stream)
