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

    def compute(self, symmetric_double_encoder, params):

        regularization = 0

        for layer in symmetric_double_encoder:

            OutputLog().write('Adding orthonormal regularization for layer')

            Wy_Square = Tensor.dot(layer.Wy.T, layer.Wy)
            Wx_Square = Tensor.dot(layer.Wx.T, layer.Wx)

            regularization += Tensor.sum((Wy_Square - Tensor.identity_like(Wy_Square)) ** 2, dtype=Tensor.config.floatX)
            regularization += Tensor.sum((Wx_Square - Tensor.identity_like(Wx_Square)) ** 2, dtype=Tensor.config.floatX)

        OutputLog().write('Computing regularization')

        regularization -= self._zeroing_param

        return regularization * (self.weight / 2) * (regularization > 0)


    def print_regularization(self, output_stream):
        super(WeightOrthonormalRegularization, self).print_regularization(output_stream)