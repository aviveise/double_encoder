from lib.Layers.symmetric_dropout_hidden_layer import SymmetricDropoutHiddenLayer
from lib.training_strategy.iterative_training_strategy import IterativeTrainingStrategy


class IterativeDropoutTrainingStrategy(IterativeTrainingStrategy):


    def _add_cross_encoder_layer(self, layer_size, symmetric_double_encoder, activation_method):

        layer_count = len(symmetric_double_encoder)

        symmetric_layer = SymmetricDropoutHiddenLayer(numpy_range=self._random_range,
                                                      hidden_layer_size=layer_size,
                                                      name="layer" + str(layer_count),
                                                      activation_hidden=activation_method,
                                                      activation_output=activation_method)

        symmetric_double_encoder.add_hidden_layer(symmetric_layer)