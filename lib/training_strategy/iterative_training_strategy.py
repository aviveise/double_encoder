__author__ = 'aviv'

from theano import shared
from lib.training_strategy.training_strategy import TrainingStrategy
from lib.stacked_double_encoder import StackedDoubleEncoder
from lib.Layers.symmetric_hidden_layer import SymmetricHiddenLayer

from lib.trainer import Trainer

class IterativeTrainingStrategy(TrainingStrategy):

    def train(self,
              training_set_x,
              training_set_y,
              hyper_parameters,
              regularization_methods,
              activation_method):

        #need to convert the input into tensor variable
        training_set_x = shared(training_set_x, 'training_set_x', borrow=True)
        training_set_y = shared(training_set_y, 'training_set_y', borrow=True)

        symmetric_double_encoder = StackedDoubleEncoder(hidden_layers=[],
                                                        numpy_range=self._random_range,
                                                        input_size=training_set_x.get_value(borrow=True).shape[1],
                                                        output_size=training_set_y.get_value(borrow=True).shape[1],
                                                        activation_method=activation_method)

        #In this phase we train the stacked encoder one layer at a time
        #once a layer was added, weights not belonging to the new layer are
        #not changed
        for layer_size in hyper_parameters.layer_sizes:

            print '--------Adding Layer of Size - %d--------\n' % layer_size
            self._add_cross_encoder_layer(layer_size, symmetric_double_encoder, activation_method)

            params = []
            params.extend(symmetric_double_encoder[-1].x_params)
            params.extend(symmetric_double_encoder[-1].y_params)

            Trainer.train(train_set_x=training_set_x,
                          train_set_y=training_set_y,
                          hyper_parameters=hyper_parameters,
                          symmetric_double_encoder=symmetric_double_encoder,
                          params=params,
                          regularization_methods=regularization_methods)

        return symmetric_double_encoder

    def _add_cross_encoder_layer(self, layer_size, symmetric_double_encoder, activation_method):

        layer_count = len(symmetric_double_encoder)

        symmetric_layer = SymmetricHiddenLayer(numpy_range=self._random_range,
                                               hidden_layer_size=layer_size,
                                               name="layer" + str(layer_count),
                                               activation_hidden=activation_method,
                                               activation_output=activation_method)

        symmetric_double_encoder.add_hidden_layer(symmetric_layer)