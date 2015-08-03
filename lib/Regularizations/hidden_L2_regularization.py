__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor


class HiddenL2Regularization(RegularizationBase):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(HiddenL2Regularization, self).__init__(regularization_parameters)
        self._layer = int(regularization_parameters['layer'])

    def compute(self, symmetric_double_encoder, params):

        regularization = 0
        if self._layer == -1:
            for layer in symmetric_double_encoder:
                hidden_x = layer.output_forward_y
                hidden_y = layer.output_forward_x

                regularization += Tensor.mean(((hidden_x - hidden_y) ** 2).sum(axis=1))

        elif self._layer < len(symmetric_double_encoder):
            hidden_x = symmetric_double_encoder[self._layer].output_forward_x
            hidden_y = symmetric_double_encoder[self._layer].output_forward_y

            regularization += Tensor.mean(((hidden_x - hidden_y) ** 2).sum(axis=1))

        return self.weight * regularization

    def print_regularization(self, output_stream):
        super(HingeRegularization, self).print_regularization(output_stream)

    def calc_hinge(self, x):

        if self._hinge_type == 'log':
            hinge = Tensor.sum(1 - Tensor.log(x ** 2 * self._k + self._b))

        elif self._hinge_type == 'abs':
            hinge = 1 - abs(x * self._k)

        elif self._hinge_type == 'sqr':
            hinge = Tensor.sum(1 - x ** 2 * self._k)

        return Tensor.sum(hinge * (hinge > 0))
