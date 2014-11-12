__author__ = 'aviv'

import numpy

from theano import tensor as Tensor
from theano import shared, config

class HiddenLayer(object):

    def __init__(self, hidden_layer_size, activation, numpy_range, name=' '):

        self.hidden_layer_size = hidden_layer_size
        self.numpy_range = numpy_range
        self.name = name

        if activation is None:
            activation = Tensor.nnet.sigmoid

        self.activation = activation

    def set_input(self, x, input_size, W=None, bias=None):

        self.x = x

        if W is None:
            initial_W = numpy.asarray(self.numpy_range.uniform(low=-4 * numpy.sqrt(6. / (self.hidden_layer_size + input_size)),
                                                               high=4 * numpy.sqrt(6. / (self.hidden_layer_size + input_size)),
                                                               size=(input_size, self.hidden_layer_size)),
                                      dtype=config.floatX)

            self.W = shared(value=initial_W, name='W' + '_' + self.name)

        if bias is None:
            self.bias = shared(value=numpy.ones(self.hidden_layer_size, dtype=config.floatX),
                               name='bias_y_prime' + '_' + self.name)

        self.params = [self.W, self.bias]
        self.output = self.activation(Tensor.dot(self.x, self.W) + self.bias)


