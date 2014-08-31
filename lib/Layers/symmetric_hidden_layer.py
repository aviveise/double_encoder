__author__ = 'aviv'

import os
import sys
import time

import numpy

import theano
import theano.tensor as Tensor
import theano.printing

class SymmetricHiddenLayer(object):

        def __init__(self, numpy_range, x=None, y=None, hidden_layer_size=0, name='',
                     activation_hidden=None, activation_output=None):

            self.hidden_layer_size = hidden_layer_size
            self.numpy_range = numpy_range
            self.name = name
            self.activation_hidden = activation_hidden
            self.activation_output = activation_output

            if self.activation_hidden is None:
                self.activation_hidden = Tensor.nnet.sigmoid

            if self.activation_output is None:
                self.activation_output = Tensor.nnet.sigmoid

            self.x = 0
            self.y = 0

            # if x is provided compute forward(x)
            if not x is None:
                self.update_x(x)

            #if y is provided compute backward(y)
            if not y is None:
                self.update_y(y)

        def update_x(self, x, weights=None, input_size=None):

            if x:

                self.x = x

                if weights is None:
                    self._initialize_input_weights(input_size)

                else:
                    self.Wx = weights


                self.bias_x = theano.shared(value=numpy.zeros(self.hidden_layer_size, dtype=theano.config.floatX),
                                                name='bias_x' + '_' + self.name)

                self.x_params = [self.Wx, self.bias_x]
                self.output_forward = self.compute_forward_hidden()

        def update_y(self, y, weights=None, output_size=None):

            #if x2 is provided compute backward(x2)
            if y:

                self.y = y

                if weights is None:
                    self._initialize_output_weights(output_size)

                else:
                    self.Wy = weights


                self.bias_y = theano.shared(value=numpy.zeros(self.hidden_layer_size, dtype=theano.config.floatX),
                                                name='bias_y' + '_' + self.name)

                self.y_params = [self.Wy, self.bias_y]
                self.output_backward = self.compute_backward_hidden()

        def _initialize_input_weights(self, input_size):

            # WXtoH is initialized with `initial_WXtoH` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU

            if input_size is None:
                raise Exception("input size not provided")

            initial_Wx = numpy.asarray(self.numpy_range.uniform(low=-4 * numpy.sqrt(6. / (self.hidden_layer_size + input_size)),
                                                                high=4 * numpy.sqrt(6. / (self.hidden_layer_size + input_size)),
                                                                size=(input_size, self.hidden_layer_size)),
                                                                dtype=theano.config.floatX)

            Wx = theano.shared(value=initial_Wx, name='Wx' + '_' + self.name)

            # WXtoH corresponds to the weights between the input and the hidden layer
            self.Wx = Wx

        def _initialize_output_weights(self, output_size):

            if output_size is None:
                raise Exception("output size not provided")

            # WHtoY is initialized with `initial_WXtoH` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_Wy = numpy.asarray(self.numpy_range.uniform(low=-4 * numpy.sqrt(6. / (self.hidden_layer_size + output_size)),
                                                                high=4 * numpy.sqrt(6. / (self.hidden_layer_size + output_size)),
                                                                size=(output_size, self.hidden_layer_size)),
                                                                dtype=theano.config.floatX)

            Wy = theano.shared(value=initial_Wy, name='Wy' + '_' + self.name)

            # WHtoY corresponds to the weights between the hidden layer and the output
            self.Wy = Wy

        def compute_forward_hidden(self):
            return self.activation_hidden(Tensor.dot(self.x, self.Wx) + self.bias_x)

        def compute_backward_hidden(self):
            return self.activation_hidden(Tensor.dot(self.y, self.Wy) + self.bias_y)

        #Given one input computes the network forward output
        def reconstruct_y(self):
            return self.activation_output(Tensor.dot(self.output_forward, self.Wy.T))

        #Given one input computes the network backward output
        def reconstruct_x(self):
            return self.activation_output(Tensor.dot(self.output_backward, self.Wx.T))

        #
        # #Regularization methods for different kinds of regularization types
        # def sparsity_forward(self, ru=0.05):
        #
        #     forward_activation = self.compute_forward_hidden()
        #
        #     avg_activation_forward = forward_activation.sum(axis=0) / forward_activation.shape[0]
        #
        #     KL_forward = ru * Tensor.log(ru / avg_activation_forward) + \
        #                    (1 - ru) * Tensor.log((1 - ru) / (1 - avg_activation_forward))
        #
        #     return KL_forward.sum()
        #
        # def sparsity_backward(self, ru=0.05):
        #
        #     backward_activation = self.compute_backward_hidden()
        #
        #     avg_activation_backward = backward_activation.sum(axis=0) / backward_activation.shape[0]
        #
        #     KL_backward = ru * Tensor.log(ru / avg_activation_backward) + \
        #                     (1 - ru) * Tensor.log((1 - ru) / (1 - avg_activation_backward))
        #
        #     return KL_backward.sum()
        #
        # def contractive_term_forward(self):
        #
        #     forward_activation = self.compute_forward_hidden()
        #
        #     h = forward_activation.sum(axis=0) / forward_activation.shape[0]
        #
        #     return ((h ** 2) * ((1 - h) ** 2) * (self.w_xtoh ** 2).sum(axis=0)).sum()
        #
        # def contractive_term_backward(self):
        #
        #     backward_activation = self.compute_backward_hidden()
        #
        #     h = backward_activation.sum(axis=0) / backward_activation.shape[0]
        #
        #     return ((h ** 2) * ((1 - h) ** 2) * (self.b_ytoh ** 2).sum(axis=0)).sum()