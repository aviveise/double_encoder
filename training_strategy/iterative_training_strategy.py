__author__ = 'aviv'

import sys
import os
import numpy

from training_strategy import TrainingStrategy
from stacked_double_encoder import StackedDoubleEncoder
from Layers.symmetric_hidden_layer import SymmetricHiddenLayer
from trainer import Trainer

class IterativeTrainingStrategy(TrainingStrategy):

    def train(self,
              training_set_x,
              training_set_y,
              hyper_parameters,
              regularization_methods,
              activation_method):

        symmetric_double_encoder = StackedDoubleEncoder(hidden_layers=[],
                                                        numpy_range=self._random_range,
                                                        activation_method=activation_method)

        #In this phase we train the stacked encoder one layer at a time
        #once a layer was added, weights not belonging to the new layer are
        #not changed
        for layer_size in hyper_parameters.layer_sizes:

            print '----------------Adding Layer - %d\n' % layer_size
            self._add_cross_encoder_layer(layer_size, symmetric_double_encoder, activation_method)

            params = []
            params.extend(self._symmetric_double_encoder[-1].x_params)
            params.extend(self._symmetric_double_encoder[-1].y_params)

            Trainer.train(self._symmetric_double_encoder, params)
            print '----------------Layer Added - %d\n' % layer_size


        return symmetric_double_encoder

    def _add_cross_encoder_layer(self, layer_size, symmetric_double_encoder, activation_method):

        layer_count = len(symmetric_double_encoder)

        symmetric_layer = SymmetricHiddenLayer(numpy_range=self._random_range,
                                               hidden_layer_size=layer_size,
                                               name="layer" + str(layer_count),
                                               activation_hidden=activation_method,
                                               activation_output=activation_method)

        symmetric_double_encoder.add_hidden_layer(symmetric_layer)