__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor

import theano.tensor.nlinalg as nlinalg


class SparseRegularization(RegularizationBase):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(SparseRegularization, self).__init__(regularization_parameters)

        self._p = float(regularization_parameters['p'])

    def compute(self, symmetric_double_encoder, params):
        regularization = 0

        for ndx, layer in enumerate(symmetric_double_encoder):
            hidden_x = layer.output_forward_y
            hidden_y = layer.output_forward_x

            p_hat_x = Tensor.abs_(hidden_x).mean(axis=0)
            kl_x = self._p * Tensor.log(self._p / p_hat_x) + (1 - self._p) * \
                                                           Tensor.log((1 - self._p) / (1 - p_hat_x))
            regularization += kl_x.sum()

            p_hat_y = Tensor.abs_(hidden_y).mean(axis=0)
            kl_y = self._p * Tensor.log(self._p / p_hat_y) + (1 - self._p) * \
                                                           Tensor.log((1 - self._p) / (1 - p_hat_y))
            regularization += kl_y.sum()

        return self.weight * regularization

    def print_regularization(self, output_stream):
        super(SparseRegularization, self).print_regularization(output_stream)
