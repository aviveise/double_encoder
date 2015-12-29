import lasagne
from math import floor
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

from lib.lasagne.learnedactivations import BatchNormalizationLayer, BatchNormLayer

WEIGHT_DECAY = 0.005


def build_model(var_x, input_size_x, var_y, input_size_y, layer_sizes,
                weight_init=lasagne.init.GlorotUniform(), drop_prob=None, **kwargs):
    hooks = {}

    # Create x to y network
    model_x, hidden_x, weights_x, biases_x, prediction_y, hooks_x = build_single_channel(var_x, input_size_x,
                                                                                         input_size_y, layer_sizes,
                                                                                         weight_init,
                                                                                         lasagne.init.Constant(0.),
                                                                                         drop_prob, 'x')

    model_y, hidden_y, weights_y, biases_y, prediction_x, hooks_y = build_single_channel(var_y, input_size_y,
                                                                                         input_size_x, layer_sizes,
                                                                                         [w.T for w in
                                                                                          reversed(weights_x)],
                                                                                         list(reversed(biases_x))[
                                                                                         1:] + [
                                                                                             lasagne.init.Constant(0.)],
                                                                                         drop_prob, 'y')

    reversed_hidden_y = list(reversed(hidden_y))

    # Merge the two dictionaries
    hooks = hooks_x
    hooks["BatchNormalizationLayer:movingavg"].extend(hooks_y["BatchNormalizationLayer:movingavg"])

    loss_x = lasagne.objectives.squared_error(var_x, prediction_x).sum(axis=1).mean()
    loss_y = lasagne.objectives.squared_error(var_y, prediction_y).sum(axis=1).mean()

    middle_layer = int(floor(float(len(hidden_x)) / 2.))

    hooks_temp = {}

    middle_x = lasagne.layers.get_output(hidden_x[middle_layer], moving_avg_hooks=hooks_temp)
    middle_y = lasagne.layers.get_output(reversed_hidden_y[middle_layer], moving_avg_hooks=hooks_temp)

    loss_l2 = lasagne.objectives.squared_error(middle_x, middle_y).sum(axis=1).mean()

    loss_weight_decay = 0

    loss_weight_decay += lasagne.regularization.regularize_layer_params(model_x,
                                                                        penalty=lasagne.regularization.l2) * WEIGHT_DECAY
    loss_weight_decay += lasagne.regularization.regularize_layer_params(model_y,
                                                                        penalty=lasagne.regularization.l2) * WEIGHT_DECAY

    loss = loss_x + loss_y + loss_l2 + loss_weight_decay

    output = {
        'loss_x': loss_x,
        'loss_y': loss_y,
        'loss_l2': loss_l2,
        'loss_weight_decay': loss_weight_decay
    }

    return model_x, model_y, hidden_x, reversed_hidden_y, loss, output, hooks


def build_single_channel(var, input_size, output_size, layer_sizes, weight_init=lasagne.init.GlorotUniform(),
                         bias_init=lasagne.init.Constant(0.), drop_prob=None, name=''):
    model = []
    weights = []
    biases = []
    hidden = []
    hooks = {}

    if isinstance(weight_init, lasagne.init.Initializer):
        weight_init = [weight_init for i in range(len(layer_sizes) + 1)]

    if isinstance(bias_init, lasagne.init.Initializer):
        bias_init = [bias_init for i in range(len(layer_sizes) + 1)]

    # Add Input Layer
    model.append(lasagne.layers.InputLayer((None, input_size), var, 'input_layer_{0}'.format(name)))

    # Add hidden layers
    for index, layer_size in enumerate(layer_sizes):
        model.append(lasagne.layers.DenseLayer(incoming=model[-1],
                                               num_units=layer_size,
                                               W=weight_init[index],
                                               b=bias_init[index],
                                               nonlinearity=lasagne.nonlinearities.LeakyRectify(0.3)))

        weights.append(model[-1].W)
        biases.append(model[-1].b)

        model.append(BatchNormalizationLayer(model[-1],
                                             nonlinearity=lasagne.nonlinearities.identity))

        drop = 0 if drop_prob is None else drop_prob[index]
        model.append(lasagne.layers.DropoutLayer(model[-1], rescale=True, p=drop))

        hidden.append(model[-1])

    # Add output layer
    model.append(lasagne.layers.DenseLayer(model[-1],
                                           num_units=output_size,
                                           W=weight_init[-1],
                                           b=bias_init[-1],
                                           nonlinearity=lasagne.nonlinearities.identity))
    weights.append(model[-1].W)
    biases.append(model[-1].b)

    prediction = lasagne.layers.get_output(model[-1], moving_avg_hooks=hooks)

    return model, hidden, weights, biases, prediction, hooks
