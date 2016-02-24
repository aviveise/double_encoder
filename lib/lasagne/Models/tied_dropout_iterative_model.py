import lasagne
from math import floor
from lasagne.layers import DenseLayer
from lasagne.regularization import l2, l1
from theano import tensor as T
from lib.lasagne.Layers.LocallyDenseLayer import TiedDenseLayer, LocallyDenseLayer
from lib.lasagne.Layers.Penelties import orthonormality
from lib.lasagne.Layers.TiedDropoutLayer import TiedDropoutLayer
from lib.lasagne.Layers.TiedNoiseLayer import TiedGaussianNoiseLayer
from lasagne.layers.noise import DropoutLayer
from lib.lasagne.learnedactivations import BatchNormalizationLayer, BatchNormLayer, BatchWhiteningLayer
from lib.lasagne.params import Params


def transpose_recursive(w):
    if not isinstance(w, list):
        return w.T

    return [transpose_recursive(item) for item in w]


def build_model(var_x, input_size_x, var_y, input_size_y, layer_sizes,
                weight_init=lasagne.init.GlorotUniform(), drop_prob=None, **kwargs):
    layer_types = Params.LAYER_TYPES

    # Create x to y network
    model_x, hidden_x, weights_x, biases_x, prediction_y, hooks_x, dropouts_x = build_single_channel(var_x,
                                                                                                     input_size_x,
                                                                                                     input_size_y,
                                                                                                     layer_sizes,
                                                                                                     layer_types,
                                                                                                     weight_init,
                                                                                                     lasagne.init.Constant(
                                                                                                         0.),
                                                                                                     drop_prob, 'x')

    weights_y = [transpose_recursive(w) for w in reversed(weights_x)]
    bias_y = lasagne.init.Constant(0.)

    model_y, hidden_y, weights_y, biases_y, prediction_x, hooks_y, dropouts_y = build_single_channel(var_y,
                                                                                                     input_size_y,
                                                                                                     input_size_x,
                                                                                                     list(reversed(
                                                                                                         layer_sizes)),
                                                                                                     list(reversed(
                                                                                                         layer_types)),
                                                                                                     weights_y,
                                                                                                     bias_y,
                                                                                                     drop_prob, 'y',
                                                                                                     dropouts_x)

    reversed_hidden_y = list(reversed(hidden_y))

    hooks = {}
    if "BatchNormalizationLayer:movingavg" in hooks_x:
        # Merge the two dictionaries
        hooks = hooks_x
        hooks["BatchNormalizationLayer:movingavg"].extend(hooks_y["BatchNormalizationLayer:movingavg"])
        # hooks["WhiteningLayer:movingavg"].extend(hooks_y["WhiteningLayer:movingavg"])

    loss_x = Params.LOSS_X * lasagne.objectives.squared_error(var_x, prediction_x).sum(axis=1).mean()
    loss_y = Params.LOSS_Y * lasagne.objectives.squared_error(var_y, prediction_y).sum(axis=1).mean()

    if len(hidden_x) % 2 == 0:
        middle_layer = int(len(hidden_x) / 2.) - 1
    else:
        middle_layer = int(floor(float(len(hidden_x)) / 2.))

    hooks_temp = {}

    layer_x = lasagne.layers.get_output(hidden_x[Params.TEST_LAYER], moving_avg_hooks=hooks_temp)
    layer_y = lasagne.layers.get_output(reversed_hidden_y[Params.TEST_LAYER], moving_avg_hooks=hooks_temp)

    loss_l2 = Params.L2_LOSS * lasagne.objectives.squared_error(layer_x, layer_y).sum(axis=1).mean()

    loss_weight_decay = 0

    shrinkage = Params.SHRINKAGE

    cov_x = T.dot(layer_x.T, layer_x) / T.cast(layer_x.shape[0], dtype=T.config.floatX)
    cov_y = T.dot(layer_y.T, layer_y) / T.cast(layer_x.shape[0], dtype=T.config.floatX)

    # mu_x = T.nlinalg.trace(cov_x) / layer_x.shape[1]
    # mu_y = T.nlinalg.trace(cov_y) / layer_y.shape[1]

    # cov_x = (1. - shrinkage) * cov_x + shrinkage * mu_x * T.identity_like(cov_x)
    # cov_y = (1. - shrinkage) * cov_y + shrinkage * mu_y * T.identity_like(cov_y)

    # loss_withen_x = Params.WITHEN_REG_X * T.mean(T.sum(abs(cov_x - T.identity_like(cov_x)), axis=0))
    # loss_withen_y = Params.WITHEN_REG_Y * T.mean(T.sum(abs(cov_y - T.identity_like(cov_y)), axis=0))

    loss_withen_x = Params.WITHEN_REG_X * (T.sqrt(T.sum(T.sum(cov_x ** 2))) - T.sqrt(T.sum(T.diag(cov_x) ** 2)))
    loss_withen_y = Params.WITHEN_REG_Y * (T.sqrt(T.sum(T.sum(cov_y ** 2))) - T.sqrt(T.sum(T.diag(cov_y) ** 2)))

    loss_weight_decay += lasagne.regularization.regularize_layer_params(model_x,
                                                                        penalty=l2) * Params.WEIGHT_DECAY
    loss_weight_decay += lasagne.regularization.regularize_layer_params(model_y,
                                                                        penalty=l2) * Params.WEIGHT_DECAY

    loss = loss_x + loss_y + loss_l2 + loss_weight_decay + loss_withen_x + loss_withen_y

    output = {
        'loss_x': loss_x,
        'loss_y': loss_y,
        'loss_l2': loss_l2,
        'loss_weight_decay': loss_weight_decay,
        'loss_withen_x': loss_withen_x,
        'loss_withen_y': loss_withen_y,
        'mean_x': T.mean(T.mean(layer_x, axis=0)),
        'mean_y': T.mean(T.mean(layer_y, axis=0)),
        'var_x': T.mean(T.var(layer_x, axis=0)),
        'var_y': T.mean(T.var(layer_y, axis=0)),
        'var_mean_x': T.var(T.mean(layer_x, axis=0)),
        'var_mean_y': T.var(T.mean(layer_y, axis=0))
    }

    return model_x, model_y, hidden_x, reversed_hidden_y, loss, output, hooks


def add_withening_regularization(hidden_x, hidden_y_reversed):
    hooks_temp = {}
    loss_withen = T.constant(0)
    for x, y in zip(hidden_x, hidden_y_reversed):
        x_value = lasagne.layers.get_output(x, moving_avg_hooks=hooks_temp)
        y_value = lasagne.layers.get_output(y, moving_avg_hooks=hooks_temp)

        cov_x = T.dot(x_value.T, x_value) / x_value.shape[0]
        cov_y = T.dot(y_value.T, y_value) / y_value.shape[0]

        loss_withen += Params.WITHEN_REG_X * T.mean(T.sum(abs(cov_x - T.identity_like(cov_x)), axis=0))
        loss_withen += Params.WITHEN_REG_Y * T.mean(T.sum(abs(cov_y - T.identity_like(cov_y)), axis=0))
    return loss_withen


def build_single_channel(var, input_size, output_size, layer_sizes, layer_types,
                         weight_init=lasagne.init.GlorotUniform(),
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
        model.append(layer_types[index](incoming=model[-1],
                                        num_units=layer_size,
                                        W=weight_init[index],
                                        b=bias_init[index],
                                        nonlinearity=lasagne.nonlinearities.LeakyRectify(Params.LEAKINESS),
                                        cell_num=Params.CELL_NUM))

        weights.append(model[-1].W)
        biases.append(model[-1].b)

        model.append(BatchNormalizationLayer(model[-1],
                                             nonlinearity=lasagne.nonlinearities.identity))

        drop = 0 if drop_prob is None else drop_prob[index]
        model.append(Params.NOISE_LAYER(model[-1], rescale=True, p=drop, noise_layer=dropouts_init[-(index + 1)]))

        dropouts.append(model[-1])

        hidden.append(model[-1])

    # Add output layer
    model.append(layer_types[-1](model[-1],
                                 num_units=output_size,
                                 W=weight_init[-1],
                                 b=bias_init[-1],
                                 nonlinearity=lasagne.nonlinearities.identity,
                                 cell_num=Params.CELL_NUM))
    weights.append(model[-1].W)
    biases.append(model[-1].b)

    prediction = lasagne.layers.get_output(model[-1], moving_avg_hooks=hooks)

    return model, hidden, weights, biases, prediction, hooks, dropouts
