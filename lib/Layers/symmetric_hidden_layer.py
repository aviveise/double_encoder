from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from MISC.logger import OutputLog

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
                 Wx=None,
                 Wy=None,
                 biasX=None,
                 biasY=None,
                 bias_primeX=None,
                 bias_primeY=None,
                 normalize=True,
                 drop=None,#'dropout',
                 k=750,
                 epsilon=1e-8,
                 dropout_prob=0.5):

        self._dropout = drop
        self._drop_probability = dropout_prob
        self.hidden_layer_size = hidden_layer_size
        self.name = name
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.normalize = normalize
        self.epsilon = epsilon
        self.is_training = is_training
        self._eval = False
        self.moving_average = []
        self._random_streams = RandomStreams()
        self._k = k
        self.gamma_x = None
        self.beta_x = None
        self.gamma_y = None
        self.beta_y = None

        if normalize:
            OutputLog().write('Using batch normalization')

        if drop:
            OutputLog().write('Using dropout')

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
            self.update_x(x, Wxn, biasX, bias_primeX)

        # if y is provided compute backward(y)
        if not y is None:
            self.update_y(y, Wy, biasY, bias_primeY)

        self.bias = theano.shared(value=numpy.zeros(self.hidden_layer_size, dtype=theano.config.floatX),
                                  name='bias' + '_' + self.name)

    def update_x(self, x, weights=None, bias=None, bias_x_prime=None, input_size=None):

        if x:

            self.x = x

            if bias:
                self.bias = bias

            if weights is None:
                self._initialize_input_weights(input_size)
            else:
                self.Wx = weights

            if bias_x_prime is None:
                self.bias_x_prime = theano.shared(value=numpy.zeros(input_size, dtype=theano.config.floatX),
                                                  name='bias_x_prime' + '_' + self.name)
            else:
                self.bias_x_prime = bias_x_prime

            if self.beta_x is None or self.gamma_x is None:
                self.beta_x = theano.shared(numpy.zeros(self.hidden_layer_size, dtype=theano.config.floatX), name='beta_x' + '_' + self.name)
                self.gamma_x = theano.shared(
                    numpy.ones(self.hidden_layer_size, dtype=theano.config.floatX),
                    #numpy.cast[theano.config.floatX](numpy.random.uniform(-0.05, 0.05, self.hidden_layer_size)),
                    name='gamma_x' + '_' + self.name)

            self.x_params = [self.Wx,
                             self.bias,
                             self.bias_x_prime,
                             self.beta_x,
                             self.gamma_x]

            self.x_hidden_params = [self.Wx, self.bias, self.beta_x, self.gamma_x]

            self.output_forward_x = self.compute_forward_hidden_x()

    def update_y(self, y, weights=None, bias_y=None, bias_y_prime=None, input_size=None, generate_weights=True):

        # if x2 is provided compute backward(x2)
        if y:

            self.y = y

            if not generate_weights:
                return

            if weights is None:
                self._initialize_output_weights(input_size)

            else:
                self.Wy = weights

            if bias_y_prime is None:
                self.bias_y_prime = theano.shared(value=numpy.zeros(input_size, dtype=theano.config.floatX),
                                                  name='bias_y_prime' + '_' + self.name)
            else:
                self.bias_y_prime = bias_y_prime

            if self.beta_y is None or  self.gamma_y is None:
                self.beta_y = theano.shared(numpy.zeros(self.hidden_layer_size, dtype=theano.config.floatX), name='beta_y' + '_' + self.name)
                self.gamma_y = theano.shared(
                    numpy.ones(self.hidden_layer_size, dtype=theano.config.floatX),
                    #numpy.cast[theano.config.floatX](numpy.random.uniform(-0.05, 0.05, self.hidden_layer_size)),
                    name='gamma_y' + '_' + self.name)

            self.y_params = [self.Wy,
                             self.bias,
                             self.bias_y_prime,
                             self.beta_y,
                             self.gamma_y]

            self.y_hidden_params = [self.Wy, self.bias, self.beta_y, self.gamma_y]

            self.output_forward_y = self.compute_forward_hidden_y()

    def _initialize_input_weights(self, input_size):

        # WXtoH is initialized with `initial_WXtoH` which is uniformely sampled
        # from -4*sqrt(6./(n_visible+n_hidden)) and
        # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
        # converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU

        if input_size is None:
            raise Exception("input size not provided")

        # wx = numpy.asarray(numpy.random.uniform(low=-numpy.sqrt(1. / input_size),
        #                                         high=numpy.sqrt(1. / input_size),
        #                                         size=(input_size, self.hidden_layer_size)),
        #                    dtype=theano.config.floatX)

        wx = numpy.random.normal(0, 0.01, size=(input_size, self.hidden_layer_size))
        OutputLog().write('Initialized Wx to normal 0 and 0.01')
        # wx = 0.01 * numpy.random.multivariate_normal(numpy.zeros(self.hidden_layer_size),
        #                                              numpy.identity(self.hidden_layer_size),
        #                                              size=input_size)
        # # wx_out = numpy.random.normal(0, 0.01, size=(self.hidden_layer_size, input_size))
        # wx = numpy.random.randn(input_size, self.hidden_layer_size) * sqrt(2.0 / input_size)

        # wx_out = wx_out.dot(scipy.linalg.inv(scipy.linalg.sqrtm(wx_out.T.dot(wx_out))))
        # wx = wx.dot(scipy.linalg.inv(scipy.linalg.sqrtm(wx.T.dot(wx))))

        initial_Wx = numpy.asarray(wx, dtype=theano.config.floatX)
        # initial_Wx_out = numpy.asarray(wx_out, dtype=theano.config.floatX)

        # self.gamma_x = theano.shared(
        #     value=numpy.random.uniform(0.95, 1.05, input_size).astype(dtype=theano.config.floatX),
        #     name='gamma_x_' + self.name)
        #
        # self.beta_x = theano.shared(value=numpy.zeros(input_size, dtype=theano.config.floatX),
        #                             name='beta_x_' + self.name)

        # WXtoH corresponds to the weights between the input and the hidden layer
        self.Wx = theano.shared(value=initial_Wx, name='Wx' + '_' + self.name)
        # self.Wx_out = theano.shared(value=initial_Wx_out, name='Wx_out' + '_' + self.name)

    def _initialize_output_weights(self, input_size):

        if input_size is None:
            raise Exception("output size not provided")

        # wy = numpy.asarray(numpy.random.uniform(low=-numpy.sqrt(1. / input_size),
        #                                         high=numpy.sqrt(1. / input_size),
        #                                         size=(input_size, self.hidden_layer_size)),
        #                    dtype=theano.config.floatX)

        OutputLog().write('Initialized Wy to normal 0 and 0.01')
        # wy = 0.01 * numpy.random.multivariate_normal(numpy.zeros(self.hidden_layer_size),
        #                                              numpy.identity(self.hidden_layer_size),
        #                                              size=input_size)
        wy = numpy.random.normal(0, 0.01, size=(input_size, self.hidden_layer_size))

        # wy = numpy.random.randn(input_size, self.hidden_layer_size) * sqrt(2.0 / input_size)

        # wy = wy.dot(scipy.linalg.inv(scipy.linalg.sqrtm(wy.T.dot(wy))))
        # wy_out = wy_out.dot(scipy.linalg.inv(scipy.linalg.sqrtm(wy_out.T.dot(wy_out))))
        #
        initial_Wy = numpy.asarray(wy, dtype=theano.config.floatX)
        # initial_Wy_out = numpy.asarray(wy_out, dtype=theano.config.floatX)

        # self.gamma_y = theano.shared(
        #     value=numpy.random.uniform(0.95, 1.05, input_size).astype(dtype=theano.config.floatX),
        #     name='gamma_y_' + self.name)
        #
        # self.beta_y = theano.shared(value=numpy.zeros(input_size, dtype=theano.config.floatX),
        #                             name='beta_y_' + self.name)

        # WHtoY corresponds to the weights between the hidden layer and the output
        self.Wy = theano.shared(value=initial_Wy, name='Wy' + '_' + self.name)
        # self.Wy_out = theano.shared(value=initial_Wy_out, name='Wy_out' + '_' + self.name)

    def compute_forward_hidden_x(self):

        layer_input = self.x

        if self._dropout == 'dropconnect':
            layer_input = self.dropconnect(layer_input, self.Wx, self.bias, self._k)
        else:
            layer_input = Tensor.dot(layer_input, self.Wx) + self.bias
        result = self.activation_hidden(layer_input)

        if self._dropout == 'dropout':
            result = self.dropout(result)

        if self.normalize:
            self.moving_average_x = []
            result = self.normalize_activations(result, self.mean_inference_x, self.variance_inference_x,
                                                self.gamma_x, self.beta_x, self.moving_average_x)

        return result

    def compute_forward_hidden_y(self):

        layer_input = self.y

        if self._dropout == 'dropconnect':
            layer_input = self.dropconnect(layer_input, self.Wy, self.bias, self._k)
        else:
            layer_input = Tensor.dot(layer_input, self.Wy) + self.bias

        result = self.activation_hidden(layer_input)

        if self._dropout == 'dropout':
            result = self.dropout(result)

        if self.normalize:
            self.moving_average_y = []
            result = self.normalize_activations(result, self.mean_inference_y, self.variance_inference_y,
                                                self.gamma_y, self.beta_y, self.moving_average_y)

        return result

    def dropout(self, input):

        p = 1 - self._drop_probability

        OutputLog().write('Using dropout with p={0}'.format(p))

        output_predict = input
        output_train = self.drop(input, p) / p

        if self._eval:
            return output_predict
        else:
            return output_train

    def dropconnect(self, input, W, b, k):

        p = 1 - self._drop_probability

        OutputLog().write('Using drop connect with k={0} and p={1}'.format(k, p))

        output_train = Tensor.dot(input, self.drop(W, p)) + b

        mean_lin_output = Tensor.dot(input, p * W) + b
        variance_output = numpy.cast[theano.config.floatX](p * (1. - p)) * Tensor.dot((input * input), (W * W))
        all_samples = self._random_streams.normal(avg=mean_lin_output, std=variance_output,
                                                  size=(k, input.shape[0], self.hidden_layer_size),
                                                  dtype=theano.config.floatX)
        output_predict = all_samples.mean(axis=0)

        if self._eval:
            return output_predict
        else:
            return output_train

    def set_eval(self, eval):
        self._eval = eval

    def drop(self, input, p):
        output_train = input * self._random_streams.binomial(input.shape,
                                                             p=p,
                                                             dtype=Tensor.config.floatX)
        return output_train

    # Given one input computes the network forward output
    def reconstruct_y(self, input_x=None):

        if input_x is None:
            hidden_x = self.output_forward_x
            return self.activation_output(Tensor.dot(hidden_x, self.Wy.T) + self.bias_y_prime)
        else:
            return self.activation_output(Tensor.dot(input_x, self.Wy.T) + self.bias_y_prime)

    # Given one input computes the network backward output
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

    def normalize_activations(self, x, mean_inference, variance_inference, gamma, beta, moving_average):

        if not self._eval:
            mean = Tensor.mean(x, axis=0, keepdims=True)
            var = Tensor.var(x, axis=0, keepdims=True)

            moving_average.append([mean, var])
            moving_average.append([mean_inference, variance_inference])

        else:
            mean = mean_inference
            var = variance_inference

        normalized_output = (x - mean) / Tensor.sqrt(var + self.epsilon)
        return normalized_output * gamma + beta
