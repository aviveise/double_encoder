import scipy
from theano.ifelse import ifelse
from theano import printing

__author__ = 'aviv'

import numpy

import theano
import theano.tensor as Tensor
import theano.printing

class SymmetricHiddenLayer(object):

        def __init__(self,
                     x=None,
                     y=None,
                     is_training=None,
                     hidden_layer_size=0,
                     name='',
                     activation_hidden=None,
                     activation_output=None,
                     Wx = None,
                     Wy = None,
                     biasX = None,
                     biasY = None,
                     bias_primeX = None,
                     bias_primeY = None,
                     normalize=False,
                     epsilon=0,
                     moving_average=None):

            self.hidden_layer_size = hidden_layer_size
            self.name = name
            self.activation_hidden = activation_hidden
            self.activation_output = activation_output
            self.normalize = normalize
            self.epsilon = epsilon
            self.is_training = is_training
            self._eval = False
            self._moving_average = moving_average

            self.mean_inference_x = theano.shared(
                numpy.zeros((1, self.hidden_layer_size), dtype=theano.config.floatX),
                borrow=True,
                broadcastable=(True, False))
            self.mean_inference_x.name = self.name + "_mean_x"

            self.variance_inference_x = theano.shared(
                numpy.zeros((1, self.hidden_layer_size), dtype=theano.config.floatX),
                borrow=True,
                broadcastable=(True, False))
            self.variance_inference_x.name = self.name + "_var_x"

            self.mean_inference_y = theano.shared(
                numpy.zeros((1, self.hidden_layer_size), dtype=theano.config.floatX),
                borrow=True,
                broadcastable=(True, False))
            self.mean_inference_y.name = self.name + "_mean_y"

            self.variance_inference_y = theano.shared(
                numpy.zeros((1, self.hidden_layer_size), dtype=theano.config.floatX),
                borrow=True,
                broadcastable=(True, False))
            self.variance_inference_y.name = self.name + "_var_y"

            if self.activation_hidden is None or self.activation_output is None:
                raise Exception('Activation must be provided')

            self.x = 0
            self.y = 0

            # if x is provided compute forward(x)
            if not x is None:
                self.update_x(x, Wx, biasX, bias_primeX)

            #if y is provided compute backward(y)
            if not y is None:
                self.update_y(y, Wy, biasY, bias_primeY)

        def update_x(self, x, weights=None, bias_x=None, bias_x_prime=None, input_size=None):

            if x:

                self.x = x

                if weights is None:
                    self._initialize_input_weights(input_size)

                else:
                    self.Wx = weights

                if bias_x is None:
                    self.bias_x = theano.shared(value=numpy.zeros(self.hidden_layer_size, dtype=theano.config.floatX),
                                                name='bias_x' + '_' + self.name)
                else:
                    self.bias_x = bias_x

                if bias_x_prime is None:
                    self.bias_x_prime = theano.shared(value=numpy.zeros(input_size, dtype=theano.config.floatX),
                                                      name='bias_x_prime' + '_' + self.name)
                else:
                    self.bias_x_prime = bias_x_prime


                self.x_params = [self.Wx]#, self.bias_x]#, self.gamma_x, self.beta_x]#, self.bias_x_prime]
                self.x_hidden_params = [self.Wx]#, self.bias_x]#, self.gamma_x, self.beta_x]

                self.output_forward_x = self.compute_forward_hidden_x()

        def update_y(self, y, weights=None, bias_y=None, bias_y_prime=None, input_size=None, generate_weights=True):

            #if x2 is provided compute backward(x2)
            if y:

                self.y = y

                if not generate_weights:
                    return

                if weights is None:
                    self._initialize_output_weights(input_size)

                else:
                    self.Wy = weights

                if bias_y is None:
                    self.bias_y = theano.shared(value=numpy.zeros(self.hidden_layer_size, dtype=theano.config.floatX),
                                                name='bias_y' + '_' + self.name)
                else:
                    self.bias_y = bias_y

                if bias_y_prime is None:
                    self.bias_y_prime = theano.shared(value=numpy.zeros(input_size, dtype=theano.config.floatX),
                                                      name='bias_y_prime' + '_' + self.name)
                else:
                    self.bias_y_prime = bias_y_prime

                self.y_params = [self.Wy]#, self.bias_y]#, self.beta_y, self.gamma_y]#, self.bias_y_prime]
                self.y_hidden_params = [self.Wy]#, self.bias_y]#, self.beta_y, self.gamma_y]

                self.output_forward_y = self.compute_forward_hidden_y()

        def _initialize_input_weights(self, input_size):

            # WXtoH is initialized with `initial_WXtoH` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU

            if input_size is None:
                raise Exception("input size not provided")

            wx = numpy.asarray(numpy.random.uniform(low=-numpy.sqrt(1. / (input_size)),
                                                            high=numpy.sqrt(1. / (input_size)),
                                                            size=(input_size, self.hidden_layer_size)),
                                                            dtype=theano.config.floatX)

            #wx = numpy.random.normal(0, 0.1, size=(input_size, self.hidden_layer_size))

            orth_wx = wx.dot(scipy.linalg.inv(scipy.linalg.sqrtm(wx.T.dot(wx))))

            initial_Wx = numpy.asarray(orth_wx, dtype=theano.config.floatX)

            self.gamma_x = theano.shared(value=numpy.random.uniform(0.95, 1.05, input_size).astype(dtype=theano.config.floatX),
                                                 name='gamma_x_' + self.name)


            self.beta_x = theano.shared(value=numpy.zeros(input_size, dtype=theano.config.floatX),
                                         name='beta_x_' + self.name)


            # WXtoH corresponds to the weights between the input and the hidden layer
            self.Wx = theano.shared(value=initial_Wx, name='Wx' + '_' + self.name)

        def _initialize_output_weights(self, input_size):

            if input_size is None:
                raise Exception("output size not provided")

            wy = numpy.asarray(numpy.random.uniform(low=-numpy.sqrt(1. / (input_size)),
                                                           high=numpy.sqrt(1. / (input_size)),
                                                            size=(input_size, self.hidden_layer_size)),
                                                            dtype=theano.config.floatX)

            #wy = numpy.random.normal(0, 0.1, size=(input_size, self.hidden_layer_size))

            orth_wy = wy.dot(scipy.linalg.inv(scipy.linalg.sqrtm(wy.T.dot(wy))))

            initial_Wy = numpy.asarray(orth_wy, dtype=theano.config.floatX)

            self.gamma_y = theano.shared(value=numpy.random.uniform(0.95, 1.05, input_size).astype(dtype=theano.config.floatX),
                             name='gamma_y_' + self.name)


            self.beta_y = theano.shared(value=numpy.zeros(input_size, dtype=theano.config.floatX),
                                         name='beta_y_' + self.name)


            # WHtoY corresponds to the weights between the hidden layer and the output
            self.Wy = theano.shared(value=initial_Wy, name='Wy' + '_' + self.name)

        def compute_forward_hidden_x(self):

            layer_input = self.x

            layer_input = Tensor.dot(layer_input, self.Wx) + self.bias_x
            result = self.activation_hidden(layer_input)

            if self.normalize:
                result = self.normalize_activations(result, self.mean_inference_x, self.variance_inference_x)

            return result

        def compute_forward_hidden_y(self):

            layer_input = self.y

            layer_input = Tensor.dot(layer_input, self.Wy) + self.bias_y
            result = self.activation_hidden(layer_input)

            if self.normalize:
                result = self.normalize_activations(result, self.mean_inference_y, self.variance_inference_y)

            return result

        def set_eval(self, eval):
            self._eval = eval


        #Given one input computes the network forward output
        def reconstruct_y(self, input_x=None):

            if input_x is None:
                hidden_x = self.output_forward_x
                return self.activation_output(Tensor.dot(hidden_x, self.Wy.T) + self.bias_y_prime)
            else:
                return self.activation_output(Tensor.dot(input_x, self.Wy.T) + self.bias_y_prime)

        #Given one input computes the network backward output
        def reconstruct_x(self, input_y=None):

            if input_y is None:
                hidden_y = self.output_forward_y
                return self.activation_output(Tensor.dot(hidden_y, self.Wx.T) + self.bias_x_prime)
            else:
                return self.activation_output(Tensor.dot(input_y, self.Wx.T) + self.bias_x_prime)

        def input_x(self):
            return self.x

        def input_y(self):
            return self.y

        def normalize_activations(self, x, mean_inference, variance_inference):

            if not self._eval:
                mean = Tensor.mean(x, axis=0, keepdims=True)
                var = Tensor.sqrt(Tensor.var(x, axis=0, keepdims=True) + self.epsilon)

                self._moving_average.append([[mean, var], [mean_inference, variance_inference]])

            else:
                mean = mean_inference
                var = variance_inference

            return (x - mean) / var