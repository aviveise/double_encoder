from math import floor, ceil
import pickle
import numpy
import scipy
import sys

__author__ = 'avive'


class DoubleEncoderTransform():
    """
    This class transforms two nxs matrices to correlated forms
    """

    @staticmethod
    def scaled_tanh(x):
        return 1.7159 * numpy.tanh(0.66 * x)

    class encodeLayer():
        def __init__(self, W, bias):
            self._W = W
            self._bias = bias


        def transform(self, x):
            """
            Transforms X to correlated subspace,
            X must be of the form SxN,
            S - samples
            N - variables
            """
            return DoubleEncoderTransform.scaled_tanh(numpy.dot(x, self._W) + self._bias)

    def __init__(self, network_path, layer_num=-1):
        """
        Constructor which imports the network
        :param network_path: path to a .mat file of the network
        :param layer_num: if -1 takes the middle layers otherwise takes the requested layer
        """
        layers_x, layers_y = self._import_encoder(network_path)

        self._layers_x = layers_x
        self._layers_y = layers_y

        if layer_num == -1:
            layer_num = int(floor(len(self._layers_x) / 2.))

        self._layer_num = layer_num

    def transform(self, x, y):
        """
        Transforms x,y to correlated forms
        :param x: matrix of size NxS where N-number of samples and S-number of features
        :param y: matrix of size NxS where N-number of samples and S-number of features
        :returns a tuple (x',y') of the transformed x and y
        """
        current_x = x
        current_y = y

        for index, layer_x in enumerate(self._layers_x):
            current_x = layer_x.transform(current_x)

            if index == self._layer_num:
                break

        for index, layer_y in enumerate(reversed(self._layers_y)):
            current_y = layer_y.transform(current_y)

            if index == (len(self._layers_y) - self._layer_num - 1):
                break

        return current_x, current_y

    def _import_encoder(self, file_name):

        encoder = scipy.io.loadmat(file_name)

        layer_number = encoder['layer_number']

        print 'importing %d layers' % layer_number

        layers_x = []
        layers_y = []

        for i in range(layer_number):

            layer_name = 'layer' + str(i)

            Wx = encoder['Wx_' + layer_name]
            bias = encoder['bias_' + layer_name]

            layer_x = DoubleEncoderTransform.encodeLayer(Wx, bias)
            layers_x.append(layer_x)


        for i in range(layer_number):

            layer_name = 'layer' + str(i)

            wy_name = 'Wy' + '_' + layer_name

            if wy_name in encoder:
                Wy = encoder[wy_name]
            else:
                Wy = layers_x[i + 1]._W.T

            layer_y = DoubleEncoderTransform.encodeLayer(Wy, layers_x[i]._bias)

            layers_y.append(layer_y)

        return layers_x, layers_y

