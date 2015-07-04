__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor

import theano.tensor.nlinalg as nlinalg

class OrthRegularization(RegularizationBase):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(OrthRegularization, self).__init__(regularization_parameters)

    def compute(self, symmetric_double_encoder, params):

        regularization = 0

        layer_number = len(symmetric_double_encoder)

        for ndx, layer in enumerate(symmetric_double_encoder):

            hidden_x = layer.output_forward_y
            hidden_y = layer.output_forward_x

            cov_x = Tensor.dot(hidden_x.T, hidden_x)
            cov_y = Tensor.dot(hidden_y.T, hidden_y)

            gama = (ndx / layer_number)

            regularization += gama * 0.5 * nlinalg.trace(cov_x - Tensor.identity_like(cov_x))
            regularization += (1 - gama) * 0.5 * nlinalg.trace(cov_y - Tensor.identity_like(cov_y))

        return regularization

    def print_regularization(self, output_stream):
        super(OrthRegularization, self).print_regularization(output_stream)