import numpy
import theano
from theano.tensor import Tensor
from theano.tensor.shared_randomstreams import RandomStreams

from lib.Layers.symmetric_hidden_layer import SymmetricHiddenLayer


class MultiVerseDoubleEncoder:

    def __init__(self,
                 input_size_x,
                 input_size_y,
                 batch_size):

        self.var_x = theano.shared(numpy.zeros((batch_size, input_size_x), dtype=Tensor.config.floatX),
                                   name='var_x')

        self.var_y = theano.shared(numpy.zeros((batch_size, input_size_y), dtype=Tensor.config.floatX),
                                   name='var_y')

        self._layers = []
        self._random_range = RandomStreams()

    def insert_layer(self, layer_size, activation_input, activation_output, verse_num):
        verse = []
        for i in range(verse_num):
            verse.append(SymmetricHiddenLayer(numpy_range=self._random_range,
                                              hidden_layer_size=layer_size,
                                              name="layer{0}_verse{1}".format(len(self._layers), verse_num),
                                              activation_hidden=activation_input,
                                              activation_output=activation_output))
        self._layers.append(verse)
