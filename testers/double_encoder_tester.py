__author__ = 'aviv'

import sys
import os
import theano.tensor as Tensor

from tester_base import TesterBase
from theano import function



class DoubleEncoderTester(TesterBase):

    def __init__(self, double_encoder, layer_num):
        super(DoubleEncoderTester, self).__init__(double_encoder)

        self._layer_num = layer_num

        #Training inputs x1 and x2 as a matrices with columns as samples
        self._x = Tensor.matrix('x')
        self._y = Tensor.matrix('y')

    def compute_outputs(self, test_set_x, test_set_y):

        model = self._build_model()
        return model(test_set_x.T, test_set_y.T)

    def _build_model(self):

        x1_tilde = self.cross_encoder_layers[self._layer_num].compute_forward_hidden()
        x2_tilde = self.cross_encoder_layers[self._layer_num].compute_backward_hidden()

        correlation_test_model = function([self._x, self._y], [x1_tilde, x2_tilde])

        return correlation_test_model


