from stacked_auto_encoder_2 import StackedDoubleEncoder2

__author__ = 'aviv'

from training_strategy import TrainingStrategy
from stacked_double_encoder import StackedDoubleEncoder
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
              validation_set_y=None,
              dir_name=None,
              encoder_type='typeA',
              import_net=False,
              import_path=''):

        if not import_net:
            symmetric_double_encoder = StackedDoubleEncoder(hidden_layers=[],
                                                            numpy_range=self._random_range,
                                                            input_size_x=training_set_x.shape[1],
                                                            input_size_y=training_set_y.shape[1],
                                                            batch_size=hyper_parameters.batch_size,
                                                            activation_method=activation_method)

        else:

            symmetric_double_encoder = StackedDoubleEncoder(hidden_layers=[],
                                                          numpy_range=self._random_range,
                                                          input_size_x=training_set_x.shape[1],
                                                          input_size_y=training_set_y.shape[1],
                                                          batch_size=hyper_parameters.batch_size,
                                                          activation_method=None)

            symmetric_double_encoder.import_encoder(import_path, hyper_parameters)



        #In this phase we train the stacked encoder one layer at a time
        #once a layer was added, weights not belonging to the new layer are
        #not changed

        layer_sizes = hyper_parameters.layer_sizes[len(symmetric_double_encoder):]

        for idx, layer_size in enumerate(layer_sizes):

            print '--------Adding Layer of Size - %d--------\n' % layer_size
            self._add_cross_encoder_layer(layer_size,
                                          symmetric_double_encoder,
                                          hyper_parameters.method_in,
                                          hyper_parameters.method_out)



            params = []

            if idx == 0:
                params.extend(symmetric_double_encoder[0].x_params)

            else:
                params.extend(symmetric_double_encoder[-1].x_hidden_params)

            params.extend(symmetric_double_encoder[-1].y_params)

            hyper_parameters.

            print '--------Starting Training Network-------\n'
            Trainer.train(train_set_x=training_set_x,
                          train_set_y=training_set_y,
                          hyper_parameters=hyper_parameters,
                          symmetric_double_encoder=symmetric_double_encoder,
                          params=params,
                          regularization_methods=regularization_methods,
                          print_verbose=print_verbose,
                          top=top,
                          validation_set_x=validation_set_x,
                          validation_set_y=validation_set_y)

            if dir_name is not None:
                symmetric_double_encoder.export_encoder(dir_name, 'layer_{0}'.format(len(symmetric_double_encoder) + 1))

        params = symmetric_double_encoder.getParams()
        hyper_parameters.learning_rate *= 0.01

        print '--------Starting Training Network-------\n'
        Trainer.train(train_set_x=training_set_x,
                          train_set_y=training_set_y,
                          hyper_parameters=hyper_parameters,
                          symmetric_double_encoder=symmetric_double_encoder,
                          params=params,
                          regularization_methods=regularization_methods,
                          print_verbose=print_verbose,
                          top=top,
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


