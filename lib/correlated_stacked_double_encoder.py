__author__ = 'aviv'


from stacked_double_encoder import StackedDoubleEncoder
from theano import tensor as Tensor
from Layers.hidden_layer import HiddenLayer

class CorrelatedStackedDoubleEncoder(StackedDoubleEncoder):

    def __init__(self, hidden_layers, numpy_range, input_size_x, input_size_y, activation_method=Tensor.nnet.sigmoid, top=50):
        super(CorrelatedStackedDoubleEncoder, self).__init__(hidden_layers, numpy_range, input_size_x,input_size_y,activation_method,0)

        self.top = top
        #self.hidden_layer_x = HiddenLayer(top, lambda x: x * (x > 0), self.numpy_range, name='output_layer_x')
        #self.hidden_layer_y = HiddenLayer(top, lambda x: x * (x > 0), self.numpy_range, name='output_layer_y')

        self.hidden_layer_x = HiddenLayer(top, Tensor.nnet.sigmoid, self.numpy_range, name='output_layer_x')
        self.hidden_layer_y = HiddenLayer(top, Tensor.nnet.sigmoid, self.numpy_range, name='output_layer_y')

    def add_hidden_layer(self, symmetric_layer):
        super(CorrelatedStackedDoubleEncoder, self).add_hidden_layer(symmetric_layer)

        layer_num = len(self._symmetric_layers)

        output_layer = int(round(layer_num / 2))

        layer = self._symmetric_layers[output_layer]

        self.hidden_layer_x.set_input(layer.output_backward, layer.hidden_layer_size)
        self.hidden_layer_y.set_input(layer.output_forward, layer.hidden_layer_size)

        self.output_x = self.hidden_layer_x.output
        self.output_y = self.hidden_layer_y.output

        self.output_params = []
        self.output_params.extend(self.hidden_layer_x.params)
        self.output_params.extend(self.hidden_layer_y.params)


