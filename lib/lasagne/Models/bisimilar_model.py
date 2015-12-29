import lasagne
from math import floor
from lib.lasagne.learnedactivations import BatchNormalizationLayer, BatchNormLayer

SIMILARITY_WEIGHT = 0.05
WEIGHT_DECAY = 0.05

def build_model(var_x, input_size_x, var_y, input_size_y, layer_sizes,
                weight_init=lasagne.init.GlorotUniform()):
    model_x = []
    model_y = []
    hidden_x = []
    hidden_y = []

    weights_x = []
    weights_y = []
    shared_biases = []

    model_x.append(lasagne.layers.InputLayer((None, input_size_x), var_x, 'input_layer_x'))
    model_y.append(lasagne.layers.InputLayer((None, input_size_y), var_y, 'input_layer_y'))

    hooks = {}

    # Create x to y network
    for index, layer_size in enumerate(layer_sizes):
        model_x.append(lasagne.layers.DenseLayer(incoming=model_x[-1],
                                                 num_units=layer_size,
                                                 W=weight_init,
                                                 b=lasagne.init.Constant(0.),
                                                 nonlinearity=lasagne.nonlinearities.rectify))

        weights_x.append(model_x[-1].W)
        shared_biases.append(model_x[-1].b)

        model_x.append(BatchNormalizationLayer(model_x[-1], nonlinearity=lasagne.nonlinearities.identity))
        # model_x.append(lasagne.layers.DropoutLayer(model_x[-1], rescale=True))

        hidden_x.append(model_x[-1])

    model_x.append(lasagne.layers.DenseLayer(model_x[-1],
                                             num_units=input_size_y,
                                             W=weight_init,
                                             b=lasagne.init.Constant(0.),
                                             nonlinearity=lasagne.nonlinearities.identity))
    shared_biases.append(model_x[-1].b)

    model_y.append(lasagne.layers.DenseLayer(incoming=model_y[-1],
                                         num_units=layer_sizes[0],
                                         W=model_x[-1].W.T,
                                         b=shared_biases[-1],
                                         nonlinearity=lasagne.nonlinearities.rectify))


    # Create y to x network
    for index, layer_size in enumerate(reversed(layer_sizes[1:])):

        model_y.append(lasagne.layers.DenseLayer(incoming=model_y[-1],
                                         num_units=layer_size,
                                         W=weights_x[-(index + 1)].W.T,
                                         b=shared_biases[-(index + 2)],
                                         nonlinearity=lasagne.nonlinearities.rectify))

        weights_y.append(model_y[-1].W)

        model_y.append(BatchNormalizationLayer(model_y[-1], nonlinearity=lasagne.nonlinearities.identity))
        # model_y.append(lasagne.layers.DropoutLayer(model_y[-1], rescale=True))

        hidden_y.append(model_y[-1])

    model_y.append(lasagne.layers.DenseLayer(model_y[-1],
                                             num_units=input_size_x,
                                             W=model_x[1].W.T,
                                             b=lasagne.init.Constant(0.),
                                             nonlinearity=lasagne.nonlinearities.identity))

    hidden_y = list(reversed(hidden_y))

    prediction_x = lasagne.layers.get_output(model_y[-1], moving_avg_hooks=hooks)
    prediction_y = lasagne.layers.get_output(model_x[-1], moving_avg_hooks=hooks)

    loss_x = lasagne.objectives.squared_error(var_x, prediction_x).sum(axis=1).mean()
    loss_y = lasagne.objectives.squared_error(var_y, prediction_y).sum(axis=1).mean()

    middle_layer = int(floor(float(len(hidden_x)) / 2.))

    hooks_temp = {}

    middle_x = lasagne.layers.get_output(hidden_x[middle_layer], moving_avg_hooks=hooks_temp)
    middle_y = lasagne.layers.get_output(hidden_y[middle_layer], moving_avg_hooks=hooks_temp)

    loss_l2 = lasagne.objectives.squared_error(middle_x, middle_y).sum(axis=1).mean()

    loss_weight_decay = 0

    loss_weight_decay += lasagne.regularization.regularize_layer_params(model_x,
                                                                        penalty=lasagne.regularization.l2) * WEIGHT_DECAY
    loss_weight_decay += lasagne.regularization.regularize_layer_params(model_y,
                                                                        penalty=lasagne.regularization.l2) * WEIGHT_DECAY


    loss = loss_x + loss_y + loss_l2 + loss_weight_decay

    bisimilar_loss = 0

    for wx, wy in zip(weights_x, reversed(weights_y)):
        bisimilar_loss += SIMILARITY_WEIGHT * lasagne.objectives.squared_error(wx, wy.T).sum()

    loss += bisimilar_loss

    output = {
        'loss_x': loss_x,
        'loss_y': loss_y,
        'loss_l2': loss_l2,
        'loss_bi': bisimilar_loss
    }

    return model_x, model_y, hidden_x, hidden_y, loss, output, hooks
