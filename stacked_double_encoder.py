__author__ = 'aviv'

import os
import sys
import time

import theano.tensor as T

from symmetric_hidden_layer import SymmetricHiddenLayer
from numpy.random import RandomState
from correlation_optimizer import CorrelationOptimizer

class StackedDoubleEncoder(object):
    def __init__(self, x, y, hidden_layers, numpy_range, activation_method=T.nnet.sigmoid):

        self._x = x
        self._y = y

        self._symmetric_layers = []

        if numpy_range is None:
            numpy_range = RandomState()

        if hidden_layers is None or hidden_layers.count() == 0:
            return

        layer_index = 0
        for layer in hidden_layers:

            input_x = None

            if layer_index == 0:
                input_x = self._x

            symmetric_layer = SymmetricHiddenLayer(numpy_rng=numpy_range,
                                                   x=input_x,
                                                   hidden_layers_size=layer,
                                                   name='layer' + layer_index,
                                                   activation_hidden=activation_method)
            self.add_hidden_layer(symmetric_layer)

            layer_index += 1

        self._symmetric_layers[-1].update_y(self._y)


    def __iter__(self):
        return self._symmetric_layers.__iter__()

    def __getitem__(self, y):
        return self._symmetric_layers.__getitem__(y)

    def add_hidden_layer(self, symmetric_layer):

        if self._symmetric_layers.count == 0:

            self._initialize_first_layer(symmetric_layer)

        else:

            last_layer = self._symmetric_layers[-1]

            #connecting the X of new layer with the Y of the last layer
            symmetric_layer.update_x(last_layer.output_y, last_layer.Wy, last_layer.bias_y)

            Wy = symmetric_layer.Wx
            bias_y = symmetric_layer.bias_x
            input_y = symmetric_layer.output_x

            #refreshing the connection between Y and X of the other layers
            for layer in reversed(self._symmetric_layers):
                layer.update_y(input_y, Wy, bias_y)

                input_y = layer.output_x
                Wy = layer.Wx
                bias_y = layer.bias_x

        #adding the new layer to the list
        self._symmetric_layers.append(symmetric_layer)

    def reconstruct_x(self):
        return self._symmetric_layers[0].reconstruct_x()

    def reconstruct_y(self):
        return self._symmetric_layers[-1].reconstruct_y()

    def _initialize_first_layer(self, layer):
        layer.update_x(self._x)
        layer.update_y(self._y)
