import itertools

import lasagne
from theano import tensor

from lib.lasagne.Models import iterative_model

ORTHOGONALITY_WEIGHT = 0.005


def orthogonal_regularization(w_list_a, w_list_b):
    loss = 0
    for wa, wb in zip(w_list_a, w_list_b):
        loss += abs(tensor.dot(wa.T, wb)).sum()

    return loss


class CompositeLayer(lasagne.layers.Layer):
    def __init__(self, layers):
        self.input_layers = layers

    @property
    def output_shape(self):
        return self.get_output_shape_for(self.input_shape)

    def get_params(self, **tags):
        return []
        # param_list = [layer.get_params(**tags) for layer in self._layers]
        # return list(itertools.chain(*param_list))

    def get_output_shape_for(self, input_shape):
        """
        Computes the output shape of this layer, given an input shape.

        Parameters
        ----------
        input_shape : tuple
            A tuple representing the shape of the input. The tuple should have
            as many elements as there are input dimensions, and the elements
            should be integers or `None`.

        Returns
        -------
        tuple
            A tuple representing the shape of the output of this layer. The
            tuple has as many elements as there are output dimensions, and the
            elements are all either integers or `None`.

        Notes
        -----
        This method will typically be overridden when implementing a new
        :class:`Layer` class. By default it simply returns the input
        shape. This means that a layer that does not modify the shape
        (e.g. because it applies an elementwise operation) does not need
        to override this method.
        """
        output_shapes = [layer.get_output_shape_for for layer in self.input_layers]

        return output_shapes[0]

    def get_output_for(self, input, **kwargs):
        raise NotImplementedError

    def add_param(self, spec, shape, name=None, **tags):
        raise NotImplementedError


def merge_wieghts(paralell_hidden_x, paralell_hidden_y, layer):

    for index, (hidden_x, hidden_y) in enumerate(zip(paralell_hidden_x, paralell_hidden_y)):
        if layer == index + 1:
            continue

        hidden_y.W = hidden_x.W.T


def build_model(var_x, input_size_x, var_y, input_size_y, layer_sizes, parallel_width, layer,
                weight_init=lasagne.init.GlorotUniform()):
    parallel_hooks = {}

    paralell_model_x = []
    paralell_model_y = []
    paralell_hidden_x = []
    paralell_hidden_y = []

    parallel_losses = []

    for i in range(parallel_width):
        model_x, model_y, hidden_x, hidden_y, loss, outputs, hooks = iterative_model.build_model(var_x,
                                                                                                 input_size_x,
                                                                                                 var_y,
                                                                                                 input_size_y,
                                                                                                 layer_sizes,
                                                                                                 weight_init)

        paralell_model_x.append(model_x)
        paralell_model_y.append(model_y)

        parallel_losses.append(loss)

        parallel_hooks.update(hooks)

        paralell_hidden_x.append(hidden_x)
        paralell_hidden_y.append(hidden_y)

    merge_wieghts(paralell_hidden_x, paralell_hidden_y, layer)

    if not layer == 0:
        model_y[-1].W = model_x[1].W.T

    if not layer == len(paralell_hidden_x) + 2:
        model_y[1].W = model_x[-1].W.T

    loss = sum(parallel_losses)

    hidden_x = list(itertools.chain(*paralell_hidden_x))
    hidden_y = list(itertools.chain(*paralell_hidden_y))

    model_x = CompositeLayer([parallel_model[-1] for parallel_model in paralell_model_x])
    model_y = CompositeLayer([parallel_model[-1] for parallel_model in paralell_model_y])

    Wx = [lasagne.layers.get_all_params(model[-1], regularizable=True) for model in paralell_model_x]

    for i in range(parallel_width):
        for j in range(i + 1, parallel_width):
            loss += ORTHOGONALITY_WEIGHT * orthogonal_regularization(Wx[i], Wx[j])

    return model_x, model_y, hidden_x, hidden_y, loss, {}, parallel_hooks
