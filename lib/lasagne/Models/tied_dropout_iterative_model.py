import lasagne
from math import floor

from lasagne.regularization import l2
from theano import tensor as T


from lib.lasagne.Layers.Penelties import orthonormality
from lib.lasagne.Layers.TiedDropoutLayer import TiedDropoutLayer
from lib.lasagne.learnedactivations import BatchNormalizationLayer, BatchNormLayer, BatchWhiteningLayer

WEIGHT_DECAY = 0.005
WITHEN_REG = 0.1
L2_LOSS = 1
LOSS_X = 1
LOSS_Y = 1

def build_model(var_x, input_size_x, var_y, input_size_y, layer_sizes,
                weight_init=lasagne.init.GlorotUniform(), drop_prob=None, **kwargs):

    # Create x to y network
    model_x, hidden_x, weights_x, biases_x, prediction_y, hooks_x, dropouts_x = build_single_channel(var_x, input_size_x,
                                                                                                   input_size_y,
                                                                                                   layer_sizes,
                                                                                                   weight_init,
                                                                                                   lasagne.init.Constant(
                                                                                                       0.),
                                                                                                   drop_prob, 'x')

    model_y, hidden_y, weights_y, biases_y, prediction_x, hooks_y, dropouts_y = build_single_channel(var_y, input_size_y,
                                                                                         input_size_x, reversed(layer_sizes),
                                                                                         [w.T for w in
                                                                                          reversed(weights_x)],
                                                                                         list(reversed(biases_x))[
                                                                                         1:] + [
                                                                                             lasagne.init.Constant(0.)],
                                                                                         drop_prob, 'y',
                                                                                         dropouts_x)

    reversed_hidden_y = list(reversed(hidden_y))

    # Merge the two dictionaries
    hooks = hooks_x
    hooks["BatchNormalizationLayer:movingavg"].extend(hooks_y["BatchNormalizationLayer:movingavg"])

    loss_x = LOSS_X * lasagne.objectives.squared_error(var_x, prediction_x).sum(axis=0).mean()
    loss_y = LOSS_Y * lasagne.objectives.squared_error(var_y, prediction_y).sum(axis=0).mean()

    middle_layer = int(floor(float(len(hidden_x)) / 2.))

    hooks_temp = {}

    middle_x = lasagne.layers.get_output(hidden_x[middle_layer], moving_avg_hooks=hooks_temp)
    middle_y = lasagne.layers.get_output(reversed_hidden_y[middle_layer], moving_avg_hooks=hooks_temp)

    loss_l2 = L2_LOSS * lasagne.objectives.squared_error(middle_x, middle_y).sum(axis=1).mean()

    loss_weight_decay = 0

    loss_withen = T.constant(0)

    cov_x = T.dot(middle_x.T, middle_x)
    cov_y = T.dot(middle_y.T, middle_y)

    loss_withen += WITHEN_REG * T.mean(T.sum(abs(cov_x - T.identity_like(cov_x)), axis=0))
    loss_withen += WITHEN_REG * T.mean(T.sum(abs(cov_y - T.identity_like(cov_y)), axis=0))

    loss_weight_decay += lasagne.regularization.regularize_layer_params(model_x,
                                                                        penalty=l2) * WEIGHT_DECAY
    loss_weight_decay += lasagne.regularization.regularize_layer_params(model_y,
                                                                        penalty=l2) * WEIGHT_DECAY

    loss = loss_x + loss_y + loss_l2 + loss_weight_decay + loss_withen

    output = {
        'loss_x': loss_x,
        'loss_y': loss_y,
        'loss_l2': loss_l2,
        'loss_weight_decay': loss_weight_decay,
        'loss_withen': loss_withen
    }

    return model_x, model_y, hidden_x, reversed_hidden_y, loss, output, hooks

def add_withening_regularization(hidden_x, hidden_y_reversed):
    hooks_temp = {}
    loss_withen = T.constant(0)
    for x, y in zip(hidden_x, hidden_y_reversed):
        x_value = lasagne.layers.get_output(x, moving_avg_hooks=hooks_temp)
        y_value = lasagne.layers.get_output(y, moving_avg_hooks=hooks_temp)

        cov_x = T.dot(x_value.T, x_value)
        cov_y = T.dot(y_value.T, y_value)

        loss_withen += WITHEN_REG * T.mean(T.sum(abs(cov_x - T.identity_like(cov_x)), axis=0))
        loss_withen += WITHEN_REG * T.mean(T.sum(abs(cov_y - T.identity_like(cov_y)), axis=0))
    return loss_withen

def build_single_channel(var, input_size, output_size, layer_sizes, weight_init=lasagne.init.GlorotUniform(),
                         bias_init=lasagne.init.Constant(0.), drop_prob=None, name='', dropouts_init=None):
    model = []
    weights = []
    biases = []
    hidden = []
    dropouts = []
    hooks = {}

    if isinstance(weight_init, lasagne.init.Initializer):
        weight_init = [weight_init for i in range(len(layer_sizes) + 1)]

    if isinstance(bias_init, lasagne.init.Initializer):
        bias_init = [bias_init for i in range(len(layer_sizes) + 1)]

    if dropouts_init is None:
        dropouts_init = [dropouts_init for i in range(len(layer_sizes) + 1)]

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
        model.append(TiedDropoutLayer(model[-1], rescale=True, p=drop, dropout_layer=dropouts_init[-(index + 1)]))

        dropouts.append(model[-1])
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

    return model, hidden, weights, biases, prediction, hooks, dropouts
