import itertools
import lasagne
from theano import tensor
from lib.lasagne.Models import iterative_model, tied_dropout_iterative_model

ORTHOGONALITY_WEIGHT = 0.005


def orthogonal_regularization(w_list_a, w_list_b):
    loss = 0
    for wa, wb in zip(w_list_a, w_list_b):
        loss += abs(tensor.dot(wa, wb.T)).sum()

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


def build_model(var_x, input_size_x, var_y, input_size_y, layer_sizes, parallel_width,
                weight_init=lasagne.init.GlorotUniform(), drop_prob=None, **kwargs):
    parallel_hooks = {"BatchNormalizationLayer:movingavg": []}

    paralell_model_x = []
    paralell_model_y = []
    paralell_hidden_x = []
    paralell_hidden_y = []

    parallel_losses = []

    for i in range(parallel_width):
        model_x, model_y, hidden_x, hidden_y, loss, outputs, hooks = tied_dropout_iterative_model.build_model(var_x,
                                                                                                              input_size_x,
                                                                                                              var_y,
                                                                                                              input_size_y,
                                                                                                              layer_sizes,
                                                                                                              weight_init,
                                                                                                              drop_prob=drop_prob)

        paralell_model_x.append(model_x)
        paralell_model_y.append(model_y)

        parallel_losses.append(loss)

        parallel_hooks["BatchNormalizationLayer:movingavg"].extend(hooks["BatchNormalizationLayer:movingavg"])

        paralell_hidden_x.append(hidden_x)
        paralell_hidden_y.append(hidden_y)

    loss = sum(parallel_losses)

    hidden_x = list(itertools.chain(*paralell_hidden_x))
    hidden_y = list(itertools.chain(*paralell_hidden_y))

    model_x = CompositeLayer([parallel_model[-1] for parallel_model in paralell_model_x])
    model_y = CompositeLayer([parallel_model[-1] for parallel_model in paralell_model_y])

    Wx = [lasagne.layers.get_all_params(model[-1], regularizable=True) for model in paralell_model_x]

    Wx = [filter(lambda param: param.name == 'W', w) for w in Wx]

    loss_orth = 0

    for i in range(parallel_width):
        for j in range(i + 1, parallel_width):
            loss_orth += ORTHOGONALITY_WEIGHT * orthogonal_regularization(Wx[i], Wx[j])

    loss += loss_orth

    output = {}
    for i in range(parallel_width):
        output['loss_{0}'.format(i)] = parallel_losses[i]

    output['loss_orth'] = loss_orth

    return model_x, model_y, hidden_x, hidden_y, loss, output, parallel_hooks
