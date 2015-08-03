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

                regularization += Tensor.mean(((hidden_x - hidden_y) ** 2).sum(axis=1, dtype=Tensor.config.floatX))

        elif self._layer < len(symmetric_double_encoder):
            hidden_x = symmetric_double_encoder[self._layer].output_forward_x
            hidden_y = symmetric_double_encoder[self._layer].output_forward_y

            regularization += Tensor.mean(
                ((hidden_x - hidden_y) ** 2).sum(axis=1, dtype=Tensor.config.floatX))

        return self.weight * regularization

    def print_regularization(self, output_stream):
        super(HiddenL2Regularization, self).print_regularization(output_stream)
