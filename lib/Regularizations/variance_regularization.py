__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor
from theano.tensor import nlinalg

class VarianceRegularization(RegularizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(VarianceRegularization, self).__init__(regularization_parameters)
        self._layer = int(regularization_parameters['layer'])

    def compute(self, symmetric_double_encoder, params):

        regularization = 0
        if self._layer == -1:
            for layer in symmetric_double_encoder:
                hidden_x = layer.output_forward_x
                hidden_y = layer.output_forward_y

                cov_x = Tensor.dot(hidden_x.T, hidden_x)
                cov_y = Tensor.dot(hidden_y.T, hidden_y)

                regularization += Tensor.sqrt((Tensor.sum(cov_x ** 2)) + Tensor.sqrt(Tensor.sum(cov_y ** 2)))

        elif self._layer < len(symmetric_double_encoder):
            hidden_x = symmetric_double_encoder[self._layer].output_forward_x
            hidden_y = symmetric_double_encoder[self._layer].output_forward_y

            cov_x = Tensor.dot(hidden_x.T, hidden_x)
            cov_y = Tensor.dot(hidden_y.T, hidden_y)

            regularization += (Tensor.sqrt((Tensor.sum(cov_x ** 2)) + Tensor.sqrt(Tensor.sum(cov_y ** 2))))


        return self.weight * regularization

    def print_regularization(self, output_stream):
        super(VarianceRegularization, self).print_regularization(output_stream)
