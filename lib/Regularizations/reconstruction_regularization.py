__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor, theano
from theano.tensor import shared_randomstreams
from theano import printing as Printing


class ReconstructionRegularization(RegularizationBase):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(ReconstructionRegularization, self).__init__(regularization_parameters)
        self._zeroing_param = float(regularization_parameters['zeroing_param'])
        self._randomStream = shared_randomstreams.RandomStreams()
        self._recon_strategy = regularization_parameters['recon_strategy']
        self._layer = int(regularization_parameters['layer'])

    def compute(self, symmetric_double_encoder, params, eps=1e-8):
        regularization = 0

        if self._layer == -1:
            for layer in symmetric_double_encoder:
                regularization += self.add_regularization(layer)
        else:
            regularization += self.add_regularization(symmetric_double_encoder[self._layer])

        return self.weight * regularization

    def add_regularization(self, layer):
        regularization = 0

        if self._recon_strategy == 'forward':
            input_x = layer.x
            recon_x = layer.reconstruct_x()

            input_y = layer.y
            recon_y = layer.reconstruct_y()

            regularization += Tensor.mean((abs(input_x - recon_x)).sum(axis=1, dtype=Tensor.config.floatX))
            regularization += Tensor.mean((abs(input_y - recon_y)).sum(axis=1, dtype=Tensor.config.floatX))
        elif self._recon_strategy == 'backward':
            input_x = layer.x
            recon_x = Tensor.dot(layer.output_forward_x,
                                 layer.Wx.T)

            input_y = layer.y
            recon_y = Tensor.dot(layer.output_forward_y,
                                 layer.Wy.T)

            regularization += Tensor.mean((abs(input_x - recon_x)).sum(axis=1, dtype=Tensor.config.floatX))
            regularization += Tensor.mean((abs(input_y - recon_y)).sum(axis=1, dtype=Tensor.config.floatX))

        return regularization

    def print_regularization(self, output_stream):
        super(ReconstructionRegularization, self).print_regularization(output_stream)
