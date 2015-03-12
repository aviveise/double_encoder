l__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor
from theano import printing as Printing

class WeightOrthonormalRegularization(RegularizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(WeightOrthonormalRegularization, self).__init__(regularization_parameters)

    def compute(self, symmetric_double_encoder, params):

        regularization = 0

        last_Wy = symmetric_double_encoder[0].Wy

        for i in xrange(1, len(symmetric_double_encoder), 1):
            layer = symmetric_double_encoder[i]
            regularization += Tensor.sum((Tensor.dot(last_Wy, layer.Wx) -
                               Tensor.ones((layer.hidden_layer_size, layer.hidden_layer_size), dtype=Tensor.config.floatX)))

            last_Wy = layer.Wy

        first_layer = symmetric_double_encoder[0]
        last_layer = symmetric_double_encoder[-1]

        regularization += Tensor.sum((Tensor.dot(first_layer.Wx.T, first_layer.Wx) -
                           Tensor.ones((first_layer.hidden_layer_size, first_layer.hidden_layer_size), dtype=Tensor.config.floatX)))


        regularization += Tensor.sum((Tensor.dot(last_layer.Wy.T, last_layer.Wy) -
                           Tensor.ones((last_layer.hidden_layer_size, last_layer.hidden_layer_size), dtype=Tensor.config.floatX)))


        return regularization * self.weight


    def print_regularization(self, output_stream):
        super(WeightOrthonormalRegularization, self).print_regularization(output_stream)