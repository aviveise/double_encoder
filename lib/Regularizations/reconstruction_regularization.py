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
        for layer in symmetric_double_encoder:
            regularization += ((layer.input_y() - layer.reconstruct_y()) ** 2).sum()
            regularization += ((layer.input_x() - layer.reconstruct_x()) ** 2).sum()

        regularization -= self._zeroing_param

        return (self.weight / 2) * regularization * (regularization > 0)


    def print_regularization(self, output_stream):
        super(ReconstructionRegularization, self).print_regularization(output_stream)
        output_stream.write('regularization_weight_zeroing_param: %f' % self._zeroing_param)
