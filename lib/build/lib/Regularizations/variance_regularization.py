__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor


class VarianceRegularization(RegularizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(VarianceRegularization, self).__init__(regularization_parameters)

    def compute(self, symmetric_double_encoder, params):

        regularization = 0
        for layer in symmetric_double_encoder:
            hidden_x = layer.output_forward_x
            hidden_y = layer.output_forward_y

            regularization -= (Tensor.sum(hidden_x ** 2) + Tensor.sum(hidden_y ** 2))

        return self.weight * regularization * (self._zeroing_param + regularization / hidden_x.shape[0].
                                               astype(dtype=Tensor.config.floatX) > 0)

    def print_regularization(self, output_stream):
        super(VarianceRegularization, self).print_regularization(output_stream)
