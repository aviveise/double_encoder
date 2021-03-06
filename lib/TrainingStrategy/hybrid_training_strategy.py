from collections import Set
from math import floor
from theano.misc.ordered_set import OrderedSet

__author__ = 'aviv'

from training_strategy import TrainingStrategy
from stacked_double_encoder import StackedDoubleEncoder
from Layers.symmetric_hidden_layer import SymmetricHiddenLayer
from MISC.logger import OutputLog
from trainer import Trainer


class HybridTrainingStrategy(TrainingStrategy):
    def __init__(self):
        super(HybridTrainingStrategy, self).__init__()
        self.name = 'hybrid'

    def train(self,
              training_set_x,
              training_set_y,
              hyper_parameters,
              regularization_methods,
              activation_method,
              print_verbose=False,
              validation_set_x=None,
              validation_set_y=None,
              dir_name=None,
              import_net=False,
              import_path='',
              reduce_val=0):

        OutputLog().write('Using Mirror Strategy')

        if len(hyper_parameters.layer_sizes) < 2:
            raise Exception('Mirror works only on more then one layer')

        layer_sizes = hyper_parameters.layer_sizes

        if not import_net:

            symmetric_double_encoder_side_A = StackedDoubleEncoder(hidden_layers=[],
                                                                   numpy_range=self._random_range,
                                                                   input_size_x=training_set_x.shape[1],
                                                                   input_size_y=training_set_y.shape[1],
                                                                   batch_size=hyper_parameters.batch_size,
                                                                   activation_method=activation_method)

            symmetric_double_encoder_side_B = StackedDoubleEncoder(hidden_layers=[],
                                                                   numpy_range=self._random_range,
                                                                   input_size_x=training_set_x.shape[1],
                                                                   input_size_y=training_set_y.shape[1],
                                                                   batch_size=hyper_parameters.batch_size,
                                                                   activation_method=activation_method)

            self._moving_average = []

            # In this phase we train the stacked encoder one layer at a time
            # once a layer was added, weights not belonging to the new layer are
            # not changed

            for idx, layer_size in enumerate(range(int(floor(len(layer_sizes) / 2)))):

                OutputLog().write('--------Adding Layer of Size - %d to A--------' % layer_sizes[0])
                self._add_cross_encoder_layer(layer_sizes[idx],
                                              symmetric_double_encoder_side_A,
                                              hyper_parameters.method_in,
                                              hyper_parameters.method_out)

                OutputLog().write('--------Adding Layer of Size - %d to B--------' % layer_sizes[-1])
                self._add_cross_encoder_layer(layer_sizes[-(idx + 1)],
                                              symmetric_double_encoder_side_B,
                                              hyper_parameters.method_in,
                                              hyper_parameters.method_out,
                                              False)

                for regularization in regularization_methods:
                    regularization.disable()

                params_size_A = []
                params_size_B = []

                if idx == 0:
                    params_size_A.extend(symmetric_double_encoder_side_A[0].x_params)
                    params_size_B.extend(symmetric_double_encoder_side_B[-1].y_params)
                else:
                    params_size_A.extend(symmetric_double_encoder_side_A[-1].x_hidden_params)
                    params_size_B.extend(symmetric_double_encoder_side_B[0].y_hidden_params)

                #params_size_A.extend(symmetric_double_encoder_side_A[-1].y_params)
                #params_size_B.extend(symmetric_double_encoder_side_B[0].x_params)

                OutputLog().write('--------Starting Training Network A-------')
                Trainer.train(train_set_x=training_set_x,
                              train_set_y=training_set_y,
                              hyper_parameters=hyper_parameters,
                              symmetric_double_encoder=symmetric_double_encoder_side_A,
                              params=params_size_A,
                              regularization_methods=regularization_methods,
                              print_verbose=print_verbose,
                              validation_set_x=validation_set_x,
                              validation_set_y=validation_set_y,
                              moving_averages=self._moving_average,
                              reduce_val=reduce_val,
                              autoencoder_x=True)

                OutputLog().write('--------Starting Training Network B-------')
                Trainer.train(train_set_x=training_set_x,
                              train_set_y=training_set_y,
                              hyper_parameters=hyper_parameters,
                              symmetric_double_encoder=symmetric_double_encoder_side_B,
                              params=params_size_B,
                              regularization_methods=regularization_methods,
                              print_verbose=print_verbose,
                              validation_set_x=validation_set_x,
                              validation_set_y=validation_set_y,
                              moving_averages=self._moving_average,
                              reduce_val=reduce_val,
                              autoencoder_y=True)

        else:
            symmetric_double_encoder = StackedDoubleEncoder(hidden_layers=[],
                                                            numpy_range=self._random_range,
                                                            input_size_x=training_set_x.shape[1],
                                                            input_size_y=training_set_y.shape[1],
                                                            batch_size=hyper_parameters.batch_size,
                                                            activation_method=None)

            symmetric_double_encoder.import_encoder(import_path, hyper_parameters)

        params = []

        if len(layer_sizes) % 2 != 0:
            self._add_cross_encoder_layer(layer_sizes[int(floor(len(layer_sizes) / 2))],
                                          symmetric_double_encoder_side_A,
                                          hyper_parameters.method_in,
                                          hyper_parameters.method_out)

        for regularization in regularization_methods:
            regularization.reset()

        symmetric_double_encoder_side_A.add_double_encoder(symmetric_double_encoder_side_B)

        for layer in symmetric_double_encoder_side_A:
            params += layer.x_hidden_params
            params.append(layer.bias)

        params.append(symmetric_double_encoder_side_A[0].bias_x_prime)
        params.append(symmetric_double_encoder_side_A[-1].bias_y_prime)
        params.append(symmetric_double_encoder_side_A[-1].Wy)

        OutputLog().write('--------Starting Training Network-------')
        Trainer.train(train_set_x=training_set_x,
                      train_set_y=training_set_y,
                      hyper_parameters=hyper_parameters,
                      symmetric_double_encoder=symmetric_double_encoder_side_A,
                      params=params,
                      regularization_methods=regularization_methods,
                      print_verbose=print_verbose,
                      validation_set_x=validation_set_x,
                      validation_set_y=validation_set_y,
                      reduce_val=reduce_val)

        if dir_name is not None:
            symmetric_double_encoder_side_A.export_encoder(dir_name, 'layer_{0}'.format(len(symmetric_double_encoder_side_A) + 1))

        return symmetric_double_encoder_side_A

    def _add_cross_encoder_layer(self,
                                 layer_size,
                                 symmetric_double_encoder,
                                 activation_hidden,
                                 activation_output,
                                 end=True):

        layer_count = len(symmetric_double_encoder)

        symmetric_layer = SymmetricHiddenLayer(hidden_layer_size=layer_size,
                                               name="layer" + str(layer_count),
                                               activation_hidden=activation_hidden,
                                               activation_output=activation_output)

        symmetric_double_encoder.add_hidden_layer(symmetric_layer, end)

    def set_parameters(self, parameters):
        return
