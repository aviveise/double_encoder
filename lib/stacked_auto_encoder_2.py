import numpy

__author__ = 'aviv'
import os
import datetime

import scipy
import theano
import theano.tensor as Tensor
from numpy.random import RandomState

from MISC.logger import OutputLog
from Layers.symmetric_hidden_layer import SymmetricHiddenLayer


class StackedDoubleEncoder2(object):

    def __init__(self, hidden_layers, numpy_range, input_size_x, input_size_y, batch_size, activation_method=Tensor.nnet.sigmoid):

        #Define x and y variables for input
        self.var_x = theano.shared(numpy.zeros((batch_size, input_size_x), dtype=Tensor.config.floatX),
                                   name='var_x')

        self.var_y = theano.shared(numpy.zeros((batch_size, input_size_y), dtype=Tensor.config.floatX),
                                   name='var_y')

        self.input_size_x = input_size_x
        self.input_size_y = input_size_y

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
            symmetric_layer.update_x(x=last_layer.output_forward_x, input_size=last_layer.hidden_layer_size)
            symmetric_layer.update_y(y=last_layer.output_forward_y, input_size=last_layer.hidden_layer_size)

        #adding the new layer to the list
        self._symmetric_layers.append(symmetric_layer)

    def reconstruct_x(self, layer_num=0):

        reconstructed_x = None

        for index, layer in enumerate(reversed(self._symmetric_layers)):
            reconstructed_x = layer.reconstruct_x(reconstructed_x)

            if index == len(self._symmetric_layers) - layer_num - 1:
                break

        return reconstructed_x

    def reconstruct_y(self, layer_num=0):

        reconstructed_y = None

        for index, layer in enumerate(reversed(self._symmetric_layers)):
            reconstructed_y = layer.reconstruct_y(reconstructed_y)

            if index == len(self._symmetric_layers) - layer_num - 1:
                break

        return reconstructed_y

    #Initialize the inputs of the first layer to be 'x' and 'y' variables
    def _initialize_first_layer(self, layer):
        layer.update_x(self.var_x, input_size=self.input_size_x)
        layer.update_y(self.var_y, input_size=self.input_size_y)

    def getParams(self):

        params_set = []
        for layer in self._symmetric_layers:
            params_set.extend(layer.x_params)
            params_set.extend(layer.y_params)

        return params_set

    def export_encoder(self, dir_name, suffix=''):

        output = {
            'layer_number': len(self._symmetric_layers)
        }

        filename = 'double_encoder_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        if isinstance(suffix, basestring):
            filename += '_' + suffix

        filename += '.mat'

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

        print 'importing %d layers' % layer_number

        x = self.var_x
        y = self.var_y

        for i in range(layer_number):

            layer_name = 'layer' + str(i)

            Wx = theano.shared(encoder['Wx_' + layer_name],
                               name='Wx_' + layer_name,
                               borrow=True)

            Wy = theano.shared(encoder['Wy_' + layer_name],
                               name='Wy_' + layer_name,
                               borrow=True)

            bias_x = theano.shared(encoder['bias_x_' + layer_name].flatten(),
                                   name='bias_x_' + layer_name,
                                   borrow=True)

            bias_y = theano.shared(encoder['bias_y_' + layer_name].flatten(),
                                   name='bias_y_' + layer_name,
                                   borrow=True)

            bias_x_prime = theano.shared(encoder['bias_x_prime_' + layer_name].flatten(),
                                         name='bias_x_prime_' + layer_name,
                                         borrow=True)

            bias_y_prime = theano.shared(encoder['bias_y_prime_' + layer_name].flatten(),
                                         name='bias_y_prime_' + layer_name,
                                         borrow=True)

            layer_size = Wx.get_value(borrow=True).shape[1]

            symmetric_layer = SymmetricHiddenLayer(numpy_range=self.numpy_range,
                                                   hidden_layer_size=layer_size,
                                                   name=layer_name,
                                                   activation_hidden=hyperparameters.method_in,
                                                   activation_output=hyperparameters.method_out)

            #connecting the X of new layer with the Y of the last layer
            symmetric_layer.update_x(x=x,
                                     weights=Wx,
                                     bias_x=bias_x,
                                     bias_x_prime=bias_x_prime)

            symmetric_layer.update_y(y=y,
                                     weights=Wy,
                                     bias_y=bias_y,
                                     bias_y_prime=bias_y_prime)

            self._symmetric_layers.append(symmetric_layer)

            x = symmetric_layer.output_forward_x
            y = symmetric_layer.output_forward_y







