__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor

import theano.tensor.nlinalg as nlinalg


class ContractiveRegularization(RegularizationBase):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(ContractiveRegularization, self).__init__(regularization_parameters)

    def compute(self, symmetric_double_encoder, params):
        regularization = 0

        for ndx, layer in enumerate(symmetric_double_encoder):
            hidden_x = layer.output_forward_y
            hidden_y = layer.output_forward_x

            input_x = layer.x
            input_y = layer.y

            J_x = Tensor.grad(hidden_x, input_x)
            J_y = Tensor.grad(hidden_y, input_y)

            regularization += Tensor.sum(J_x ** 2)
            regularization += Tensor.sum(J_y ** 2)

        return regularization

    def print_regularization(self, output_stream):
        super(ContractiveRegularization, self).print_regularization(output_stream)
