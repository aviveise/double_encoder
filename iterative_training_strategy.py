__author__ = 'aviv'

import sys
import os
import numpy

from training_strategy import TrainingStrategy
from stacked_double_encoder import StackedDoubleEncoder
from symmetric_hidden_layer import SymmetricHiddenLayer
from theano.compat.python2x import OrderedDict
from theano import shared, config


class IterativeTrainingStrategy(TrainingStrategy):

    def __init__(self, dataset, hyper_parameters, regularization_methods, activation_method):
        super(IterativeTrainingStrategy, self).__init__(dataset, hyper_parameters)

        self._regularization_methods = regularization_methods
        self._symmetric_double_encoder = StackedDoubleEncoder(x=self._x,
                                                              y=self._y,
                                                              hidden_layers=[],
                                                              numpy_range=self._random_range,
                                                              activation_method=activation_method)


        self._activation_method = activation_method
        self._layer_count = 0

    def get_regularization_methods(self):
        return self._regularization_methods

    def start_training(self):

        #In this phase we train the stacked encoder one layer at a time
        #once a layer was added, weights not belonging to the new layer are
        #not changed
        for layer_size in self._hyper_parameters.layer_sizes:

            print '--------------Training layer %d----------------' % i

            self._add_cross_encoder_layer(layer_size)

            params = []
            params.extend(self._symmetric_double_encoder[-1].x_params)
            params.extend(self._symmetric_double_encoder[-1].y_params)

            self._trainer.train(self._symmetric_double_encoder, params)

        return self._symmetric_double_encoder

    def _add_cross_encoder_layer(self, layer_size):

        symmetric_layer = SymmetricHiddenLayer(numpy_range=self._random_range,
                                               hidden_layer_size=layer_size,
                                               name="layer" + self._layer_count,
                                               activation_hidden=self._activation_method,
                                               activation_output=self._activation_method)

        self._symmetric_double_encoder.add_hidden_layer(symmetric_layer)
        self._layer_count += 1