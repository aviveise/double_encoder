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

    def computeA(self, symmetric_double_encoder, params):

        regularization = 0
        if self._layer == -1:
            for layer in symmetric_double_encoder:
                hidden_x = layer.output_forward_x
                hidden_y = layer.output_forward_y

                cov_x = Tensor.dot(hidden_x, hidden_x.T)
                cov_y = Tensor.dot(hidden_y, hidden_y.T)

                regularization += Tensor.mean(Tensor.sum(abs(cov_x), axis=1, dtype=Tensor.config.floatX)) + Tensor.mean(
                    Tensor.sum(abs(cov_y), axis=1, dtype=Tensor.config.floatX))

        elif self._layer < len(symmetric_double_encoder):
            hidden_x = symmetric_double_encoder[self._layer].output_forward_x
            hidden_y = symmetric_double_encoder[self._layer].output_forward_y

            var_x = Tensor.var(hidden_x, axis=1)
            var_y = Tensor.var(hidden_y, axis=1)

            norm_x = Tensor.mean(Tensor.sum(hidden_x ** 2, axis=1, dtype=Tensor.config.floatX))
            norm_y = Tensor.mean(Tensor.sum(hidden_y ** 2, axis=1, dtype=Tensor.config.floatX))

            regularization -= norm_x
            regularization -= norm_y

            #
            # cov_x = Tensor.dot(hidden_x.T, hidden_x)
            # cov_y = Tensor.dot(hidden_y.T, hidden_y)
            #
            # regularization -= ((Tensor.sum(abs(cov_x))) + (Tensor.sum(abs(cov_y))))

        return self.weight * regularization

    def print_regularization(self, output_stream):
        super(VarianceRegularization, self).print_regularization(output_stream)
