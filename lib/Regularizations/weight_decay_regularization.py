__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor

class WeightDecayRegularization(RegularizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(WeightDecayRegularization, self).__init__(regularization_parameters)
        self.normal_type = regularization_parameters['weight_decay_type']

    def compute(self, symmetric_double_encoder, params):

        if self.normal_type == 'L1':
            regularization = self._compute_L1(symmetric_double_encoder, params)

        elif self.normal_type == 'L2':
            regularization = self._compute_L2(symmetric_double_encoder, params)

        else:
            raise Exception('unknown weight decay regularization type')

        return (self.weight / 2) * regularization

    def _compute_L1(self, symmetric_double_encoder):

        regularization = 0
        for layer in symmetric_double_encoder:
            regularization += layer.Wx.sum() + layer.Wy.sum()

        return regularization

    def _compute_L2(self, symmetric_double_encoder, params):

        regularization = 0
        for param in params:
            regularization += Tensor.sum(param ** 2)

        return regularization

    def print_regularization(self, output_stream):

        super(WeightDecayRegularization, self).print_regularization(output_stream)

        output_stream.write('weight_decay_type: %s' % self.normal_type)
