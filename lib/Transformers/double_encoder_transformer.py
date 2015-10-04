import cv2
from MISC.logger import OutputLog

__author__ = 'aviv'
import numpy
import math

from theano import function, theano
from transformer_base import TransformerBase


class DoubleEncoderTransformer(TransformerBase):
    def __init__(self, double_encoder, layer_num):
        super(DoubleEncoderTransformer, self).__init__(double_encoder)

        self._layer_num = layer_num

        # Training inputs x1 and x2 as a matrices with columns as samples
        self._x = self._correlation_optimizer.var_x
        self._y = self._correlation_optimizer.var_y

    def compute_outputs(self, test_set_x, test_set_y, hyperparameters):

        hidden_output_model = self._build_hidden_model()
        recon_model = self._build_reconstruction_model()

        hidden_values_x = []
        hidden_values_y = []

        number_of_batches = int(math.ceil(float(test_set_x.shape[0]) / hyperparameters.batch_size))

        outputs_hidden = None
        outputs_recon = None

        for i in range(number_of_batches):

            batch_x = test_set_x[
                      i * hyperparameters.batch_size: min((i + 1) * hyperparameters.batch_size, test_set_x.shape[0]), :]
            batch_y = test_set_y[
                      i * hyperparameters.batch_size: min((i + 1) * hyperparameters.batch_size, test_set_y.shape[0]), :]

            self._correlation_optimizer.var_x.set_value(batch_x)
            self._correlation_optimizer.var_y.set_value(batch_y)

            start_tick = cv2.getTickCount()
            outputs_hidden_batch = hidden_output_model()
            output_recon_batch = recon_model()

            tickFrequency = cv2.getTickFrequency()
            current_time = cv2.getTickCount()

            OutputLog().write('batch {0}/{1} ended, time: {2}'.format(i,
                                                                      number_of_batches,
                                                                      ((current_time - start_tick) / tickFrequency)),
                              'debug')

            if i == 0:
                outputs_recon = output_recon_batch
            else:
                for idx in range(len(outputs_recon)):
                    outputs_recon[idx] = numpy.concatenate((outputs_recon[idx], output_recon_batch[idx]), axis=0)

            if i == 0:
                outputs_hidden = outputs_hidden_batch
            else:
                for idx in range(len(outputs_hidden)):
                    outputs_hidden[idx] = numpy.concatenate((outputs_hidden[idx], outputs_hidden_batch[idx]), axis=0)

        for i in xrange(len(outputs_hidden) / 2):
            hidden_values_x.append(outputs_hidden[2 * i])
            hidden_values_y.append(outputs_hidden[2 * i + 1])

        # for i in xrange(len(outputs_recon) / 2):
        #     hidden_values_x.append(outputs_recon[2 * i])
        #     hidden_values_y.append(outputs_recon[2 * i + 1])

        # if hidden_values_x[0].shape[1] == hidden_values_y[-1].shape[1] and len(hidden_values_x) > 1:
        #     hidden_values_x.append(hidden_values_x[-1])
        #     hidden_values_y.append(hidden_values_y[0])

        return [hidden_values_x, hidden_values_y]

    def _build_hidden_model(self):

        outputs = []

        for layer in self._correlation_optimizer:
            outputs.append(layer.output_forward_x)
            outputs.append(layer.output_forward_y)

        correlation_test_model = function([], outputs)

        return correlation_test_model

    def _build_reconstruction_model(self):

        outputs = []

        outputs.append(self._correlation_optimizer.reconstruct_x())
        outputs.append(self._correlation_optimizer.var_x)

        outputs.append(self._correlation_optimizer.reconstruct_y())
        outputs.append(self._correlation_optimizer.var_y)

        correlation_test_model = function([], outputs)

        return correlation_test_model
