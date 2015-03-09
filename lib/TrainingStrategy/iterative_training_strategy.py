__author__ = 'aviv'

import numpy
from theano import tensor as Tensor
from theano import scan
from theano.tensor import nlinalg
from theano.compat.python2x import OrderedDict
from theano import function
from theano import config
from theano import shared
from theano import Out

from training_strategy import TrainingStrategy
from stacked_double_encoder import StackedDoubleEncoder
from correlated_stacked_double_encoder import CorrelatedStackedDoubleEncoder
from Layers.symmetric_hidden_layer import SymmetricHiddenLayer
from MISC.logger import OutputLog
from trainer import Trainer

class IterativeTrainingStrategy(TrainingStrategy):

    def __init__(self):
        super(IterativeTrainingStrategy, self).__init__()
        OutputLog().write('\nStrategy: Iterative')
        self.name = 'iterative'

    def train(self,
              training_set_x,
              training_set_y,
              hyper_parameters,
              regularization_methods,
              activation_method,
              top=50,
              print_verbose=False,
              validation_set_x=None,
              validation_set_y=None):

        symmetric_double_encoder = StackedDoubleEncoder(hidden_layers=[],
                                                        numpy_range=self._random_range,
                                                        input_size=training_set_x.shape[1],
                                                        output_size=training_set_y.shape[1],
                                                        activation_method=activation_method)

        #In this phase we train the stacked encoder one layer at a time
        #once a layer was added, weights not belonging to the new layer are
        #not changed
        for layer_size in hyper_parameters.layer_sizes:

            print '--------Adding Layer of Size - %d--------\n' % layer_size
            self._add_cross_encoder_layer(layer_size,
                                          symmetric_double_encoder,
                                          hyper_parameters.method_in,
                                          hyper_parameters.method_out)

            params = []
            params.extend(symmetric_double_encoder[-1].x_hidden_params)
            params.extend(symmetric_double_encoder[-1].y_params)

            Trainer.train(train_set_x=training_set_x,
                          train_set_y=training_set_y,
                          hyper_parameters=hyper_parameters,
                          symmetric_double_encoder=symmetric_double_encoder,
                          params=params,
                          regularization_methods=regularization_methods,
                          print_verbose=print_verbose,
                          validation_set_x=validation_set_x,
                          validation_set_y=validation_set_y)

        return symmetric_double_encoder


    def _add_cross_encoder_layer(self, layer_size, symmetric_double_encoder, activation_hidden, activation_output):

        layer_count = len(symmetric_double_encoder)

        symmetric_layer = SymmetricHiddenLayer(numpy_range=self._random_range,
                                               hidden_layer_size=layer_size,
                                               name="layer" + str(layer_count),
                                               activation_hidden=activation_hidden,
                                               activation_output=activation_output)

        symmetric_double_encoder.add_hidden_layer(symmetric_layer)

    def set_parameters(self, parameters):
        return

    def _compute_square_chol(self, a, n):

        w, v = Tensor.nlinalg.eigh(a,'L')

        result, updates = scan(lambda eigs, eigv, prior_results, size: Tensor.sqrt(eigs) * Tensor.dot(eigv.reshape([size, 1]), eigv.reshape([1, size])),
                               outputs_info=Tensor.zeros_like(a),
                               sequences=[w, v.T],
                               non_sequences=n)

        return result[-1]


