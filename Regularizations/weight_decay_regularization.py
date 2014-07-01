__author__ = 'aviv'

import sys
import io

from regularization_base import RegularizationBase


class WeightDecayRegularization(RegularizationBase):

    def __init__(self, weight, regularization_type='L2'):
        super(WeightDecayRegularization, self).__init__(weight)
        self.regularization_type = regularization_type

    def compute(self, symmetric_double_encoder):

        if self.regularization_type == 'L1':
            regularization = self._compute_L1(symmetric_double_encoder)

        elif self.regularization_type == 'L2':
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
            regularization += layer.Wx.square().sum() + layer.Wy.square().sum()

        return regularization