import numpy
from MISC.logger import OutputLog

__author__ = 'aviv'
import theano


from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor
from theano import printing as Printing

class WeightOrthonormalRegularization(RegularizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(WeightOrthonormalRegularization, self).__init__(regularization_parameters)
        self._weight_base = self.weight

    def disable(self):
        self.weight = 0

    def reset(self):
        self.weight = self._weight_base

    def compute(self, symmetric_double_encoder, params):

        regularization = 0

        for layer in symmetric_double_encoder:

            OutputLog().write('Adding orthonormal regularization for layer')

            # hidden_x = layer.output_forward_x
            # hidden_y = layer.output_forward_y
            #
            # hidden_x_sqr = Tensor.dot(hidden_x.T, hidden_x)
            # hidden_y_sqr = Tensor.dot(hidden_y.T, hidden_y)
            #
            # regularization += Tensor.sum((hidden_x_sqr - Tensor.identity_like(hidden_x_sqr)) ** 2, dtype=Tensor.config.floatX)
            # regularization += Tensor.sum((hidden_y_sqr - Tensor.identity_like(hidden_y_sqr)) ** 2, dtype=Tensor.config.floatX)

            Wy_Square = Tensor.dot(layer.Wy.T, layer.Wy)
            Wx_Square = Tensor.dot(layer.Wx.T, layer.Wx)
            # cross = Tensor.dot(layer.Wx, layer.Wy.T)
            #
            # regularization += Tensor.sum((Wy_Square - Tensor.identity_like(Wy_Square)) ** 2, dtype=Tensor.config.floatX)
            regularization += Tensor.sum(abs(Wx_Square), dtype=Tensor.config.floatX)
            regularization += Tensor.sum(abs(Wy_Square), dtype=Tensor.config.floatX)
            # regularization += Tensor.sum(abs(cross))

        Wy_Square = Tensor.dot(symmetric_double_encoder[-1].Wy.T, symmetric_double_encoder[-1].Wy)
        regularization += Tensor.sum(Wy_Square ** 2, dtype=Tensor.config.floatX)

        OutputLog().write('Computing regularization')

        regularization -= self._zeroing_param

        return regularization * (self.weight / 2) * (regularization > 0)


    def print_regularization(self, output_stream):
        super(WeightOrthonormalRegularization, self).print_regularization(output_stream)