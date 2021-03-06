from Layers.symmetric_dropout_hidden_layer import SymmetricDropoutHiddenLayer
from iterative_training_strategy import IterativeTrainingStrategy

from MISC.logger import OutputLog

class IterativeDropoutTrainingStrategy(IterativeTrainingStrategy):

    def __init__(self):
        super(IterativeDropoutTrainingStrategy, self).__init__()
        OutputLog().write('\nStrategy: Iterative Dropout')
        self.name = 'iterative_dropout'
        self.probability = 0.5


    def _add_cross_encoder_layer(self, layer_size, symmetric_double_encoder, activation_hidden, activation_output):

        layer_count = len(symmetric_double_encoder)

        symmetric_layer = SymmetricDropoutHiddenLayer(numpy_range=self._random_range,
                                                      hidden_layer_size=layer_size,
                                                      name="layer" + str(layer_count),
                                                      activation_hidden=activation_hidden,
                                                      activation_output=activation_output,
                                                      dropout_probability=self.probability)

        symmetric_double_encoder.add_hidden_layer(symmetric_layer)

    def set_parameters(self, parameters):

        try:
            self.probability = float(parameters[self.name]['probability'])
            print 'setting dropout probability: %f' % self.probability

        finally:
            return
