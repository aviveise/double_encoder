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

        last_input_x = test_set_x
        last_input_y = test_set_y

        hidden_output_model = self._build_hidden_model()
        reconstruction_output_model = self._build_reconstruction_model()

        hidden_values_x = []
        hidden_values_y = []
        output_values_x = []
        output_values_y = []

        outputs_hidden = hidden_output_model(test_set_x, test_set_y)
        outputs_reconstruct = reconstruction_output_model(test_set_x, test_set_y)

        for i in xrange(len(outputs_hidden) / 2):
            hidden_values_x.append(outputs_hidden[2 * i])
            hidden_values_y.append(outputs_hidden[2 * i + 1])


        for i in xrange(len(outputs_reconstruct) / 2):
            output_values_x.append(outputs_reconstruct[2 * i])
            output_values_y.append(outputs_reconstruct[2 * i + 1])

        return [hidden_values_x, hidden_values_y], [output_values_x, output_values_y]


    def _build_hidden_model(self):

        outputs = []

        for layer in self._correlation_optimizer:
            outputs.append(layer.compute_forward_hidden())
            outputs.append(layer.compute_backward_hidden())

        correlation_test_model = function([self._x, self._y], outputs)

        return correlation_test_model

    def _build_reconstruction_model(self):

        outputs = []

        for layer in self._correlation_optimizer:

            outputs.append(layer.reconstruct_x())
            outputs.append(layer.input_x())

            outputs.append(layer.reconstruct_y())
            outputs.append(layer.input_y())

        correlation_test_model = function([self._x, self._y], outputs)

        return correlation_test_model

