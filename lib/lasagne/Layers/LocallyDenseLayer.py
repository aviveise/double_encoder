import lasagne
import theano
import numpy as np
from lasagne import nonlinearities
from lasagne.layers import Layer, DenseLayer
from lasagne.random import get_rng
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from lasagne import init


class LocallyDenseLayer(Layer):
    """Dropout layer

    Sets values to zero with probability p. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If true the input is rescaled with input / (1-p) when deterministic
        is False.

    Notes
    -----
    The dropout layer is a regularizer that randomly sets input values to
    zero; see [1]_, [2]_ for why this might improve generalization.
    During training you should set deterministic to false and during
    testing you should set deterministic to true.

    If rescale is true the input is scaled with input / (1-p) when
    deterministic is false, see references for further discussion. Note that
    this implementation scales the input at training time.

    References
    ----------
    .. [1] Hinton, G., Srivastava, N., Krizhevsky, A., Sutskever, I.,
           Salakhutdinov, R. R. (2012):
           Improving neural networks by preventing co-adaptation of feature
           detectors. arXiv preprint arXiv:1207.0580.

    .. [2] Srivastava Nitish, Hinton, G., Krizhevsky, A., Sutskever,
           I., & Salakhutdinov, R. R. (2014):
           Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
           Journal of Machine Learning Research, 5(Jun)(2), 1929-1958.
    """

    def __init__(self, incoming, num_units, cell_num, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 name=None,**kwargs):
        super(LocallyDenseLayer, self).__init__(incoming, name)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))
        self.cell_input_size = num_inputs / cell_num
        self.cell_size = self.num_units / cell_num

        if isinstance(W, lasagne.init.Initializer):
            W = [W for i in range(0, cell_num)]

        if isinstance(b, lasagne.init.Initializer):
            b = [b for i in range(0, cell_num)]

        self._dense_layers = []
        self.W = []
        self.b = []
        for i in range(cell_num):
            self._dense_layers.append(TiedDenseLayer(DownsizeLayer(incoming, cell_num),
                                                     self.cell_size, W[i], b[i], nonlinearity, **kwargs))

            self.W.append(self._dense_layers[-1].W)
            self.b.append(self._dense_layers[-1].b)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        outputs = []

        for i, layer in enumerate(self._dense_layers):
            outputs.append(layer.get_output_for(input[:, self.cell_input_size * i: self.cell_input_size * (i + 1)]))

        return T.concatenate(outputs, axis=1)

    def get_params(self, **tags):
        results = []

        for layer in self._dense_layers:
            results.extend(layer.get_params(**tags))

        return results


class DownsizeLayer():
    def __init__(self, layer, downsize):
        self._layer = layer
        self.output_shape = (layer.output_shape[0], layer.output_shape[1] / downsize)


class TiedDenseLayer(DenseLayer):
    def __init__(self, incoming, num_units,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, name=None, **kwargs):
        super(TiedDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity, name=name)

        if not isinstance(W, lasagne.init.Initializer):
            self.params[self.W].remove('trainable')
            self.params[self.W].remove('regularizable')

        if self.b and not isinstance(b, lasagne.init.Initializer):
            self.params[self.b].remove('trainable')
