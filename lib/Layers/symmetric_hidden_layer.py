from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from MISC.logger import OutputLog

__author__ = 'aviv'

import numpy

import theano
import theano.tensor as Tensor
import theano.printing
from theano.tensor import nlinalg


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
                 normalize_sample=False,
                 decorrelate=False,
                 drop='dropout',
                 k=750,
                 epsilon=1e-6,
                 dropout_prob=0.5):

        self._dropout = drop
        self._drop_probability = dropout_prob
        self.hidden_layer_size = hidden_layer_size
        self.name = name
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.normalize = normalize
        self.normalize_sample = normalize_sample
        self.decorrelate = decorrelate
        self.epsilon = epsilon
        self.is_training = is_training
        self._eval = False
        self.moving_average_x = []
        self.moving_average_y = []
        self._random_streams = RandomStreams()
        self._k = k
        self.gamma_x = None
        self.beta_x = None
        self.gamma_y = None
        self.beta_y = None
        self.beta = None

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


        # self.cov_inference_x = theano.shared(
        #     numpy.cast[theano.config.floatX](self.generate_random_basis(self.hidden_layer_size, self.hidden_layer_size)),
        #     borrow=True)
        # self.variance_inference_x.name = self.name + "_cov_x"
        #
        # self.cov_inference_y = theano.shared(
        #     numpy.cast[theano.config.floatX](self.generate_random_basis(self.hidden_layer_size, self.hidden_layer_size)),
        #     borrow=True)
        # self.variance_inference_x.name = self.name + "_cov_x"

        if self.activation_hidden is None or self.activation_output is None:
            raise Exception('Activation must be provided')

        self.x = 0
        self.y = 0

        # if x is provided compute forward(x)
        if not x is None:
            self.update_x(x, Wx, biasX, bias_primeX)

        # if y is provided compute backward(y)
        if not y is None:
            self.update_y(y, Wy, biasY, bias_primeY)

        self.bias = theano.shared(value=numpy.zeros(self.hidden_layer_size, dtype=theano.config.floatX),
                                  name='bias' + '_' + self.name)

        self.beta = theano.shared(value=numpy.zeros(self.hidden_layer_size, dtype=theano.config.floatX),
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
                self.beta_x = theano.shared(numpy.zeros(self.hidden_layer_size, dtype=theano.config.floatX),
                                            name='beta_x' + '_' + self.name)
                self.gamma_x = theano.shared(
                    numpy.ones(self.hidden_layer_size, dtype=theano.config.floatX),
                    name='gamma_x' + '_' + self.name)

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

            if self.beta_y is None or self.gamma_y is None:
                self.beta_y = theano.shared(numpy.zeros(self.hidden_layer_size, dtype=theano.config.floatX),
                                            name='beta_y' + '_' + self.name)
                self.gamma_y = theano.shared(
                    numpy.ones(self.hidden_layer_size, dtype=theano.config.floatX),
                    # numpy.cast[theano.config.floatX](numpy.random.uniform(1.00, 1.05, self.hidden_layer_size)),
                    name='gamma_y' + '_' + self.name)

            self.output_forward_y = self.compute_forward_hidden_y()

    def _initialize_input_weights(self, input_size):

        # WXtoH is initialized with `initial_WXtoH` which is uniformely sampled
        # from -4*sqrt(6./(n_visible+n_hidden)) and
        # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
        # converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU

        if input_size is None:
            raise Exception("input size not provided")

        OutputLog().write('Initialized Wx to 1/input size')
        wx = numpy.asarray(numpy.random.uniform(low=-numpy.sqrt(1. / (input_size + self.hidden_layer_size)),
                                                high=numpy.sqrt(1. / (input_size + self.hidden_layer_size)),
                                                size=(input_size, self.hidden_layer_size)),
                           dtype=theano.config.floatX)

        initial_Wx = numpy.asarray(wx, dtype=theano.config.floatX)

        # WXtoH corresponds to the weights between the input and the hidden layer
        self.Wx = theano.shared(value=initial_Wx, name='Wx' + '_' + self.name)
        # self.Wx_out = theano.shared(value=initial_Wx_out, name='Wx_out' + '_' + self.name)

    def _initialize_output_weights(self, input_size):

        if input_size is None:
            raise Exception("output size not provided")

        wy = numpy.asarray(numpy.random.uniform(low=-numpy.sqrt(1. / (input_size + self.hidden_layer_size)),
                                                high=numpy.sqrt(1. / (input_size + self.hidden_layer_size)),
                                                size=(input_size, self.hidden_layer_size)),
                           dtype=theano.config.floatX)

        OutputLog().write('Initialized Wy to 1 / (input size + layer size)')

        initial_Wy = numpy.asarray(wy, dtype=theano.config.floatX)

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
        # result = layer_input

        if self.normalize:
            self.moving_average_x = []
            result = self.normalize_activations(result, self.mean_inference_x, self.variance_inference_x,
                                                self.gamma_x, self.beta, self.moving_average_x)

        if self.normalize_sample:
            result = self.normalize_samples(result, self.gamma_x, self.beta)

        if self.decorrelate:
            self.moving_average_x = []
            result = self.withen_activations(result, self.cov_inference_x, self.moving_average_x, self.gamma_x,
                                             self.bias)

        if self._dropout == 'dropout':
            result = self.dropout(result)

        # result = self.activation_hidden(result)

        return result

    def compute_forward_hidden_y(self):

        layer_input = self.y

        if self._dropout == 'dropconnect':
            layer_input = self.dropconnect(layer_input, self.Wy, self.bias, self._k)
        else:
            layer_input = Tensor.dot(layer_input, self.Wy) + self.bias

        result = self.activation_hidden(layer_input)
        # result = layer_input

        if self.normalize:
            self.moving_average_y = []
            result = self.normalize_activations(result, self.mean_inference_y, self.variance_inference_y,
                                                self.gamma_y, self.beta, self.moving_average_y)

        if self.normalize_sample:
            result = self.normalize_samples(result, self.gamma_y, self.beta)

        if self.decorrelate:
            self.moving_average_y = []
            result = self.withen_activations(result, self.cov_inference_y, self.moving_average_y, self.gamma_y,
                                             self.beta_y)

        if self._dropout == 'dropout':
            result = self.dropout(result)


        # result = self.activation_hidden(result)


        return result

    def dropout(self, input):

        p = 1 - self._drop_probability

        OutputLog().write('Using dropout with p={0}'.format(p))

        output_predict = input * p
        output_train = self.drop(input, p)

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
        output_train = input * Tensor.cast(self._random_streams.binomial(input.shape, p=p), dtype=Tensor.config.floatX)
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

    def normalize_samples(self, x, gamma, beta):
        OutputLog().write('Normalizing Samples')
        mean = Tensor.mean(x, axis=1, keepdims=True)
        var = Tensor.var(x, axis=1, keepdims=True)

        normalized_output = (x - mean) / Tensor.sqrt(var + self.epsilon)
        return normalized_output / gamma + beta

    def normalize_activations(self, x, mean_inference, variance_inference, gamma, beta, moving_average):

        if not self._eval:
            mean = Tensor.mean(x, axis=0, keepdims=True)
            var = Tensor.var(x, axis=0, keepdims=True)
            std = Tensor.sqrt(var + self.epsilon)

            # std = Tensor.std(x, axis=0, keepdims=True)

            moving_average.append([mean, std])
            moving_average.append([mean_inference, variance_inference])

        else:
            mean = mean_inference
            std = variance_inference

        # std += self.epsilon
        normalized_output = (x - mean) / std
        return normalized_output / gamma + beta

    def withen_activations(self, x, mean_cov, moving_average, gamma, beta):

        if not self._eval:
            cov = Tensor.dot(x.T, x)

            v, w = Tensor.nlinalg.eigh(cov)

            moving_average.append([w])
            moving_average.append([mean_cov])
        else:
            w = mean_cov

        decorrelated_output = Tensor.dot(x, w.T)
        return decorrelated_output * gamma + beta

    def generate_random_basis(self, n, m):
        random_vectors = []
        for i in range(m):
            vr = numpy.random.normal(0, 1, n)
            vo = numpy.copy(vr)
            for v in random_vectors:
                proj_vec = v * numpy.dot(vr, v) / numpy.dot(v, v)
                vo -= proj_vec
            random_vectors.append(vo)

        return numpy.vstack(random_vectors)
