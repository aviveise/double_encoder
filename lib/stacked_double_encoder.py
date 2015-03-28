

__author__ = 'aviv'
import os
import scipy
import datetime
import theano
import theano.tensor as Tensor

from MISC.logger import OutputLog
from numpy.random import RandomState
from Layers.symmetric_hidden_layer import SymmetricHiddenLayer

class StackedDoubleEncoder(object):

    def __init__(self, hidden_layers, numpy_range, input_size, output_size, activation_method=Tensor.nnet.sigmoid):

        #Define x and y variables for input
        self.var_x = Tensor.matrix('x')
        self.var_y = Tensor.matrix('y')

        self.input_size = input_size
        self.output_size = output_size

        self._symmetric_layers = []

        if numpy_range is None:
            numpy_range = RandomState()

        self.numpy_range = numpy_range

        if hidden_layers is None or len(hidden_layers) == 0:
            return

        layer_index = 0
        for layer in hidden_layers:

            input_x = None

            if layer_index == 0:
                input_x = self.var_x

            symmetric_layer = SymmetricHiddenLayer(numpy_rng=numpy_range,
                                                   x=input_x,
                                                   hidden_layers_size=layer,
                                                   name='layer' + layer_index,
                                                   activation_hidden=activation_method)
            self.add_hidden_layer(symmetric_layer)

            layer_index += 1

    def __iter__(self):
        return self._symmetric_layers.__iter__()

    def __getitem__(self, y):
        return self._symmetric_layers.__getitem__(y)

    def __len__(self):
        return len(self._symmetric_layers)

    def add_hidden_layer(self, symmetric_layer):

        if len(self._symmetric_layers) == 0:

            self._initialize_first_layer(symmetric_layer)

        else:

            last_layer = self._symmetric_layers[-1]

            #connecting the X of new layer with the Y of the last layer
            symmetric_layer.update_y(self.var_y, output_size=self.output_size)
            symmetric_layer.update_x(x=last_layer.output_forward, input_size=last_layer.hidden_layer_size)

            Wy = symmetric_layer.Wx.T

            input_y = symmetric_layer.output_backward

            #refreshing the connection between Y and X of the other layers
            for layer in reversed(self._symmetric_layers):

                layer.update_y(input_y, Wy, layer.bias_y, layer.bias_y_prime)

                Wy = layer.Wx.T

                input_y = layer.output_backward

        #adding the new layer to the list
        self._symmetric_layers.append(symmetric_layer)

    def reconstruct_x(self):
        return self._symmetric_layers[0].reconstruct_x()

    def reconstruct_y(self):
        return self._symmetric_layers[-1].reconstruct_y()

    #Initialize the inputs of the first layer to be 'x' and 'y' variables
    def _initialize_first_layer(self, layer):
        layer.update_x(self.var_x, input_size=self.input_size)
        layer.update_y(self.var_y, output_size=self.output_size)

    def getParams(self):

        params_set = set()
        for layer in self._symmetric_layers:
            for param in layer.y_hidden_params:
                params_set.add(param)
            for param in layer.x_hidden_params:
                params_set.add(param)

        for param in self._symmetric_layers[-1].y_params:
            params_set.add(param)
        for param in self._symmetric_layers[0].x_params:
            params_set.add(param)


        return list(params_set)

    def export_encoder(self, dir_name):

        output = {
            'layer_number': len(self._symmetric_layers)
        }

        filename = 'double_encoder_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.mat'

        for layer in self._symmetric_layers:

            for param in layer.x_params:
                OutputLog().write('exporting param:' + param.name)
                try:
                    output[param.name] = param.get_value(borrow=True)
                except:
                    OutputLog().write('Failed exporting param:' + param.name)

            for param in layer.y_params:
                OutputLog().write('exporting param:' + param.name)

                try:
                    output[param.name] = param.get_value(borrow=True)
                except:
                    OutputLog().write('Failed exporting param:' + param.name)

        scipy.io.savemat(os.path.join(dir_name, filename), output)

    def import_encoder(self, file_name, hyperparameters):

        encoder = scipy.io.loadmat(file_name)

        layer_number = encoder['layer_number']

        x = self.var_x
        y = self.var_y

        for i in range(layer_number):

            layer_name = 'layer' + str(i)

            Wx = theano.shared(encoder['Wx' + '_' + layer_name],
                               name='Wx' + '_' + layer_name,
                               borrow=True)

            bias_x = theano.shared(encoder['bias_x' + '_' + layer_name].flatten(),
                                   name='bias_x' + '_' + layer_name,
                                   borrow=True)

            bias_y = theano.shared(encoder['bias_y' + '_' + layer_name].flatten(),
                                   name='bias_y' + '_' + layer_name,
                                   borrow=True)

            bias_x_prime = theano.shared(encoder['bias_x_prime' + '_' + layer_name].flatten(),
                                         name='bias_x_prime' + '_' + layer_name,
                                         borrow=True)

            bias_y_prime = theano.shared(encoder['bias_y_prime' + '_' + layer_name].flatten(),
                                         name='bias_y_prime' + '_' + layer_name,
                                         borrow=True)

            layer_size = Wx.get_value(borrow=True).shape[1]

            layer = SymmetricHiddenLayer(numpy_range=self.numpy_range,
                                         hidden_layer_size=layer_size,
                                         name=layer_name,
                                         activation_hidden=hyperparameters.method_in,
                                         activation_output=hyperparameters.method_out)

            wy_name = 'Wy' + '_' + layer_name

            layer.update_x(x,
               Wx=Wx,
               bias_x=bias_x,
               bias_x_prime=bias_x_prime)


            if wy_name in encoder:

                Wy = theano.shared(encoder[wy_name],
                                   name='Wy' + '_' + layer_name,
                                   borrow=True)

                layer.update_y(y,
                               Wy=Wy,
                               bias_y=bias_y,
                               bias_y_prime=bias_y_prime,
                               output_size=bias_y_prime.get_value(borrow=True).shape[0])

            else:

                layer.update_y(y,
                               bias_y=bias_y,
                               bias_y_prime=bias_y_prime,
                               output_size=bias_y_prime.get_value(borrow=True).shape[0])

            self._symmetric_layers.append(layer)


        y = self._symmetric_layers[-1].output_backward
        Wy = self._symmetric_layers[-1].Wx.T

        for back_layer in reversed(self._symmetric_layers[0:-1]):
            back_layer.update_y(y, Wy, back_layer.bias_y, back_layer.bias_y_prime)
            Wy = back_layer.Wx.T
            y = back_layer.output_backward


