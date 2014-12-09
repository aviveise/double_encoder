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

        test_result_x = []
        test_result_y = []

        for layer in self._correlation_optimizer:
            model = self._build_model(layer)
            x_tilde, y_tilde = model(test_set_x, test_set_y)
            test_result_x.append(x_tilde)
            test_result_y.append(y_tilde)

        model = self._build_reconstruction_model()
        x_tilde, y_tilde = model(test_set_x, test_set_y)

        test_result_x.append(x_tilde)
        test_result_y.append(test_set_x)

        test_result_x.append(y_tilde)
        test_result_y.append(test_set_y)

        return test_result_x, test_result_y


    def _build_model(self, layer):

        x1_tilde = layer.compute_forward_hidden()
        x2_tilde = layer.compute_backward_hidden()

        correlation_test_model = function([self._x, self._y], [x1_tilde, x2_tilde])

        return correlation_test_model

    #def compute_reconstructions(self, test_x, test_y):
    #    model = self._build_reconstruction_model()
    #    return model(test_x, test_y)

    def _build_reconstruction_model(self):

        x_tilde = self._correlation_optimizer.reconstruct_x()
        y_tilde = self._correlation_optimizer.reconstruct_y()

        correlation_test_model = function([self._x, self._y], [x_tilde, y_tilde])

        return correlation_test_model

