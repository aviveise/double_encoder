__author__ = 'aviv'

from theano import function
from tester_base import TesterBase


class DoubleEncoderTester(TesterBase):

    def __init__(self, double_encoder, layer_num):
        super(DoubleEncoderTester, self).__init__(double_encoder)

        self._layer_num = layer_num

        #Training inputs x1 and x2 as a matrices with columns as samples
        self._x = self._correlation_optimizer.var_x
        self._y = self._correlation_optimizer.var_y

    def compute_outputs(self, test_set_x, test_set_y):

        model = self._build_model()
        return model(test_set_x, test_set_y)

    def _build_model(self):

        x1_tilde = self._correlation_optimizer[self._layer_num].compute_forward_hidden()
        x2_tilde = self._correlation_optimizer[self._layer_num].compute_backward_hidden()

        correlation_test_model = function([self._x, self._y], [x1_tilde, x2_tilde])

        return correlation_test_model


