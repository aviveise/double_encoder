__author__ = 'aviv'

from lib.Regularizations.regularization_base import RegularizationBase
from lib.MISC.container import ContainerRegisterMetaClass

class WeightDecayRegularization(RegularizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(WeightDecayRegularization, self).__init__(regularization_parameters)
        self.normal_type = regularization_parameters['weight_decay_type']

    def compute(self, symmetric_double_encoder):

        if self.normal_type == 'L1':
            regularization = self._compute_L1(symmetric_double_encoder)

        elif self.normal_type == 'L2':
            regularization = self._compute_L2(symmetric_double_encoder)

        else:
            raise Exception('unknown weight decay regularization type')

        return (self.weight / 2) * regularization

    def _compute_L1(self, symmetric_double_encoder):

        regularization = 0
        for layer in symmetric_double_encoder:
            regularization += layer.Wx.sum() + layer.Wy.sum()

        return regularization

    def _compute_L2(self, symmetric_double_encoder):

        regularization = 0
        for layer in symmetric_double_encoder:
            regularization += (layer.Wx ** 2).sum() + (layer.Wy ** 2).sum()

        return regularization

    def print_regularization(self, output_stream):

        super(WeightDecayRegularization, self).print_regularization(output_stream)

        output_stream.write('weight_decay_type: %s' % self.normal_type)
