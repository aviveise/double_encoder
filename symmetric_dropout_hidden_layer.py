__author__ = 'aviv'

import os
import sys
import time

import numpy

import theano
import theano.tensor as Tensor
import theano.printing

from theano.tensor.shared_randomstreams import RandomStreams

from symmetric_hidden_layer import SymmetricHiddenLayer

class SymmetricDropoutHiddenLayer(SymmetricHiddenLayer):

    def __init__(self, numpy_range, x=None, y=None, hidden_layer_size=0, name='',
                 activation_hidden=None, activation_output=None, dropout_probability=0.5):
        super(SymmetricDropoutHiddenLayer, self).__init__(numpy_range, x, y, hidden_layer_size, name,
                                                          activation_hidden=activation_hidden,
                                                          activation_output=activation_output)

        self._p = dropout_probability

    def compute_forward_hidden(self):
        output = super(SymmetricHiddenLayer, self).compute_forward_hidden()
        return self._dropout_from_layer(output)

    def compute_backward_hidden(self):
        output = super(SymmetricHiddenLayer, self).compute_backward_hidden()
        return self._dropout_from_layer(output)

    def _dropout_from_layer(self, layer, p):

        stream = RandomStreams(self.numpy_range.randint(999999))

        mask = stream.binomial(size=layer.shape, n=1, p=(1-self._p))

        return layer * Tensor.cast(mask, theano.config.floatX)