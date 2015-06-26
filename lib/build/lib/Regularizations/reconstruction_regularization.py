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

    def compute(self, symmetric_double_encoder, params, eps=1e-8):
        regularization = 0

        if self._recon_strategy == 'all':
            for index, layer in enumerate(symmetric_double_encoder):
                # regularization += Tensor.mean((layer.output_forward_y - layer.output_forward_x).norm(2, axis=1))

                mod_y = Tensor.sqrt(Tensor.sum(layer.output_forward_y ** 2, 1) + eps)
                mod_x = Tensor.sqrt(Tensor.sum(layer.output_forward_x ** 2, 1) + eps)
                regularization += 1 - Tensor.mean(
                    Tensor.diag(Tensor.dot(layer.output_forward_x, layer.output_forward_y.T)) / (mod_y * mod_x))
                #
                # perm_indx = self._randomStream.permutation(n=layer.output_forward_x.shape[0], size=(1,))
                #
                # regularization -= Tensor.cast(
                #     (layer.output_forward_y - layer.output_forward_x[perm_indx]).norm(2, axis=1)
                #         .sum(dtype=Tensor.config.floatX), dtype=Tensor.config.floatX)

        elif self._recon_strategy == 'last':
            layer = symmetric_double_encoder[-1]

            regularization += Tensor.mean((layer.output_forward_y - layer.output_forward_x).norm(2, axis=1))

            perm_indx = self._randomStream.permutation(n=layer.output_forward_x.shape[0], size=(1,))

            regularization -= Tensor.cast((layer.output_forward_y - layer.output_forward_x[perm_indx]).norm(2, axis=1)
                                          .sum(dtype=Tensor.config.floatX), dtype=Tensor.config.floatX)

        elif self._recon_strategy == 'twin':
            layer_x = symmetric_double_encoder[0]
            layer_y = symmetric_double_encoder[-1]

            regularization += Tensor.cast((layer_y.output_forward_x - layer_x.output_forward_y).norm(2, axis=1).sum(
                dtype=Tensor.config.floatX) / layer_x.output_forward_x.shape[0], dtype=Tensor.config.floatX)

            perm_indx = self._randomStream.permutation(n=layer_y.output_forward_y.shape[0], size=(1,))

            regularization -= Tensor.cast(
                (layer_y.output_forward_x - layer_x.output_forward_y[perm_indx]).norm(2, axis=1)
                    .sum(dtype=Tensor.config.floatX), dtype=Tensor.config.floatX)
        else:
            raise Exception('unknown recon command')

        return (self.weight / 2) * regularization

    def print_regularization(self, output_stream):
        super(ReconstructionRegularization, self).print_regularization(output_stream)
