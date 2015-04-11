__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor
from theano import printing as Printing

class ReconstructionRegularization(RegularizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(ReconstructionRegularization, self).__init__(regularization_parameters)
        self._zeroing_param = float(regularization_parameters['zeroing_param'])

    def compute(self, symmetric_double_encoder, params):

        regularization = 0
        for index, layer in enumerate(symmetric_double_encoder):

            if not layer.input_y() == symmetric_double_encoder.var_y:
                regularization += Tensor.sum((layer.input_y() - symmetric_double_encoder.reconstruct_y(index)),
                                            dtype=Tensor.config.floatX,
                                            acc_dtype=Tensor.config.floatX)

            if not layer.input_x() == symmetric_double_encoder.var_x:
                regularization += Tensor.sum((layer.input_x() - symmetric_double_encoder.reconstruct_x(index)),
                                             dtype=Tensor.config.floatX,
                                             acc_dtype=Tensor.config.floatX)

        regularization -= self._zeroing_param

        return (self.weight / 2) * regularization * (regularization > 0)


    def print_regularization(self, output_stream):
        super(ReconstructionRegularization, self).print_regularization(output_stream)
        output_stream.write('regularization_weight_zeroing_param: %f' % self._zeroing_param)
