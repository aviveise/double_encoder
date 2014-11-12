__author__ = 'aviv'

from theano import function
from transformer_base import TransformerBase


class DoubleEncoderTransformer(TransformerBase):

    def __init__(self, double_encoder, layer_num):
        super(DoubleEncoderTransformer, self).__init__(double_encoder)

        self._layer_num = layer_num

        #Training inputs x1 and x2 as a matrices with columns as samples
        self._x = self._correlation_optimizer.var_x
        self._y = self._correlation_optimizer.var_y

    def compute_outputs(self, test_set_x, test_set_y):

        model = self._build_model()
        return model(test_set_x, test_set_y)

    def compute_reconstructions(self, test_x, test_y):
        model = self._build_reconstruction_model()
        return model(test_x, test_y)

    def _build_model(self):

        x1_tilde = self._correlation_optimizer.output_x
        x2_tilde = self._correlation_optimizer.output_y

        correlation_test_model = function([self._x, self._y], [x1_tilde, x2_tilde])

        return correlation_test_model

    def _build_reconstruction_model(self):

        x_tilde = self._correlation_optimizer.reconstruct_x()
        y_tilde = self._correlation_optimizer.reconstruct_y()

        correlation_test_model = function([self._x, self._y], [x_tilde, y_tilde])

        return correlation_test_model

