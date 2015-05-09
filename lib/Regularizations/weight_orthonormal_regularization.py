import numpy
from MISC.logger import OutputLog

l__author__ = 'aviv'
import theano


from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor
from theano import printing as Printing

class WeightOrthonormalRegularization(RegularizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(WeightOrthonormalRegularization, self).__init__(regularization_parameters)
        self._zeroing_param = float(regularization_parameters['zeroing_param'])

    def computeA(self, symmetric_double_encoder, params):

        regularization = 0

        last_Wy = symmetric_double_encoder[0].Wy

        for i in xrange(1, len(symmetric_double_encoder), 1):
            layer = symmetric_double_encoder[i]
            regularization += Tensor.sum((Tensor.dot(last_Wy, layer.Wx) -
                               Tensor.ones((layer.hidden_layer_size, layer.hidden_layer_size), dtype=Tensor.config.floatX)),
                                         dtype=Tensor.config.floatX,
                                         acc_dtype=Tensor.config.floatX)

            last_Wy = layer.Wy

        first_layer = symmetric_double_encoder[0]
        last_layer = symmetric_double_encoder[-1]

        regularization += Tensor.sum((Tensor.dot(first_layer.Wx.T, first_layer.Wx) -
                           Tensor.ones((first_layer.hidden_layer_size, first_layer.hidden_layer_size), dtype=Tensor.config.floatX)),
                                     dtype=Tensor.config.floatX,
                                     acc_dtype=Tensor.config.floatX)


        regularization += Tensor.sum((Tensor.dot(last_layer.Wy.T, last_layer.Wy) -
                           Tensor.ones((last_layer.hidden_layer_size, last_layer.hidden_layer_size), dtype=Tensor.config.floatX)),
                                     dtype=Tensor.config.floatX,
                                     acc_dtype=Tensor.config.floatX)


        return regularization * self.weight

    def compute(self, symmetric_double_encoder, params):

        regularization = 0

        for layer in symmetric_double_encoder:

            OutputLog().write('Adding orthonormal regularization for layer')

            Wy_Square = Tensor.dot(layer.Wy.T, layer.Wy)
            Wx_Square = Tensor.dot(layer.Wx.T, layer.Wx)

            regularization += Tensor.sum(abs(Wy_Square - Tensor.identity_like(Wy_Square)), dtype=Tensor.config.floatX)
            regularization += Tensor.sum(abs(Wx_Square - Tensor.identity_like(Wx_Square)), dtype=Tensor.config.floatX)

        OutputLog().write('Computing regularization')

        return regularization * (self.weight / 2) * (regularization > 0)


    def print_regularization(self, output_stream):
        super(WeightOrthonormalRegularization, self).print_regularization(output_stream)