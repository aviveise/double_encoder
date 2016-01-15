import itertools
import lasagne
import numpy as np
import theano
import theano.tensor as T
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers import Layer

__all__ = [
    "BatchNormalizationLayer"
]


class BatchWhiteningLayer(Layer):
    def __init__(self, incoming,
                 W=init.GlorotNormal(),
                 gamma=init.Uniform([0.95, 1.05]),
                 beta=init.Constant(0.),
                 epsilon=0.001,
                 **kwargs):
        super(BatchWhiteningLayer, self).__init__(incoming, **kwargs)

        self.num_units = int(np.prod(self.input_shape[1:]))
        self.W = self.add_param(W, (self.num_units, self.num_units), name="WhiteningLayer:W", regularizable=False,
                                trainable=False)

        self.gamma = self.add_param(gamma, (self.num_units,), name="BatchNormalizationLayer:gamma", regularizable=True)
        self.beta = self.add_param(beta, (self.num_units,), name="BatchNormalizationLayer:beta", regularizable=False)
        self.epsilon = epsilon

        self.mean_inference = theano.shared(
            np.zeros((1, self.num_units), dtype=theano.config.floatX),
            borrow=True,
            broadcastable=(True, False))
        self.mean_inference.name = "shared:mean"

        self.variance_inference = theano.shared(
            np.zeros((1, self.num_units), dtype=theano.config.floatX),
            borrow=True,
            broadcastable=(True, False))
        self.variance_inference.name = "shared:variance"

        self.R_inference = theano.shared(
            np.zeros((self.num_units, self.num_units), dtype=theano.config.floatX),
            borrow=True,
            broadcastable=(False, False))
        self.variance_inference.name = "shared:R"

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, moving_avg_hooks=None,
                       deterministic=False, *args, **kwargs):

        if deterministic is False:

            m = T.mean(input, axis=0, keepdims=True)
            m.name = "tensor:mean"
            v = T.sqrt(T.var(input, axis=0, keepdims=True) + self.epsilon)
            v.name = "tensor:variance"

            R = T.dot(((input - m)).T, ((input - m)))

            key = "WhiteningLayer:movingavg"
            if key not in moving_avg_hooks:
                moving_avg_hooks[key] = []
            moving_avg_hooks[key].append(
                [[self.R_inference], [self.W]])

            key = "BatchNormalizationLayer:movingavg"
            if key not in moving_avg_hooks:
                moving_avg_hooks[key] = []
            moving_avg_hooks[key].append(
                [[m, v, R], [self.mean_inference, self.variance_inference, self.R_inference]])
        else:
            m = self.mean_inference
            v = self.variance_inference

        input_hat = T.dot((input - m), self.W.T) # normalize
        y = input_hat / self.gamma + self.beta  # scale and shift

        return y


class BatchNormalizationLayer(Layer):
    """
    Batch normalization Layer [1]
    The user is required to setup updates for the learned parameters (Gamma
    and Beta). The values nessesary for creating the updates can be
    obtained by passing a dict as the moving_avg_hooks keyword to
    get_output().

    REF:
     [1] http://arxiv.org/abs/1502.03167

    :parameters:
        - input_layer : `Layer` instance
            The layer from which this layer will obtain its input

        - nonlinearity : callable or None (default: lasagne.nonlinearities.rectify)
            The nonlinearity that is applied to the layer activations. If None
            is provided, the layer will be linear.

        - epsilon : scalar float. Stabilizing training. Setting this too
            close to zero will result in nans.
    """

    def __init__(self, incoming,
                 gamma=init.Uniform([0.95, 1.05]),
                 beta=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 epsilon=0.001,
                 **kwargs):
        super(BatchNormalizationLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = int(np.prod(self.input_shape[1:]))
        self.gamma = self.add_param(gamma, (self.num_units,), name="BatchNormalizationLayer:gamma", regularizable=True)
        self.beta = self.add_param(beta, (self.num_units,), name="BatchNormalizationLayer:beta", regularizable=False)
        self.epsilon = epsilon

        self.mean_inference = theano.shared(
            np.zeros((1, self.num_units), dtype=theano.config.floatX),
            borrow=True,
            broadcastable=(True, False))
        self.mean_inference.name = "shared:mean"

        self.variance_inference = theano.shared(
            np.zeros((1, self.num_units), dtype=theano.config.floatX),
            borrow=True,
            broadcastable=(True, False))
        self.variance_inference.name = "shared:variance"

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, moving_avg_hooks=None,
                       deterministic=False, *args, **kwargs):

        if deterministic is False:
            m = T.mean(input, axis=0, keepdims=True)
            v = T.sqrt(T.var(input, axis=0, keepdims=True) + self.epsilon)
            m.name = "tensor:mean"
            v.name = "tensor:variance"

            key = "BatchNormalizationLayer:movingavg"
            if key not in moving_avg_hooks:
                moving_avg_hooks[key] = []
            moving_avg_hooks[key].append(
                [[m, v], [self.mean_inference, self.variance_inference]])
        else:
            m = self.mean_inference
            v = self.variance_inference

        input_hat = (input - m) / v  # normalize
        y = input_hat / self.gamma + self.beta  # scale and shift

        if input.ndim > 2:
            y = T.reshape(y, output_shape)
        return self.nonlinearity(y)


# create updates
def batchnormalizeupdates(hooks, avglen):
    params = list(itertools.chain(*[i[1] for i in hooks['BatchNormalizationLayer:movingavg']]))
    tensors = list(itertools.chain(*[i[0] for i in hooks['BatchNormalizationLayer:movingavg']]))

    updates = []
    mulfac = 1.0 / avglen
    for tensor, param in zip(tensors, params):
        updates.append((param, (1.0 - mulfac) * param + mulfac * tensor))
    return updates


def whiteningupdates(hooks, learning_rate=0.0001):
    params = list(itertools.chain(*[i[1] for i in hooks["WhiteningLayer:movingavg"]]))
    Rs = list(itertools.chain(*[i[0] for i in hooks["WhiteningLayer:movingavg"]]))

    updates = []
    for tensor, param in zip(Rs, params):
        updates.append((param, param + learning_rate * (param - T.dot(tensor, param))))
    return updates


class BatchNormLayer(lasagne.layers.Layer):
    def __init__(self, incoming, axes=None, epsilon=0.01, alpha=0.5,
                 nonlinearity=None, **kwargs):
        """
        Instantiates a layer performing batch normalization of its inputs,
        following Ioffe et al. (http://arxiv.org/abs/1502.03167).

        @param incoming: `Layer` instance or expected input shape
        @param axes: int or tuple of int denoting the axes to normalize over;
            defaults to all axes except for the second if omitted (this will
            do the correct thing for dense layers and convolutional layers)
        @param epsilon: small constant added to the standard deviation before
            dividing by it, to avoid numeric problems
        @param alpha: coefficient for the exponential moving average of
            batch-wise means and standard deviations computed during training;
            the larger, the more it will depend on the last batches seen
        @param nonlinearity: nonlinearity to apply to the output (optional)
        """
        super(BatchNormLayer, self).__init__(incoming, **kwargs)
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
        dtype = theano.config.floatX
        self.mean = self.add_param(lasagne.init.Constant(0), shape, 'mean',
                                   trainable=False, regularizable=False)
        self.std = self.add_param(lasagne.init.Constant(1), shape, 'std',
                                  trainable=False, regularizable=False)
        self.beta = self.add_param(lasagne.init.Constant(0), shape, 'beta',
                                   trainable=True, regularizable=True)
        self.gamma = self.add_param(lasagne.init.Constant(1), shape, 'gamma',
                                    trainable=True, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            # use stored mean and std
            mean = self.mean
            std = self.std
        else:
            # use this batch's mean and std
            mean = input.mean(self.axes, keepdims=True)
            std = input.std(self.axes, keepdims=True)
            # and update the stored mean and std:
            # we create (memory-aliased) clones of the stored mean and std
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * mean)
            running_std.default_update = ((1 - self.alpha) * running_std +
                                          self.alpha * std)
            # and include them in the graph so their default updates will be
            # applied (although the expressions will be optimized away later)
            mean += 0 * running_mean
            std += 0 * running_std
        std += self.epsilon
        mean = T.addbroadcast(mean, *self.axes)
        std = T.addbroadcast(std, *self.axes)
        beta = T.addbroadcast(self.beta, *self.axes)
        gamma = T.addbroadcast(self.gamma, *self.axes)
        normalized = (input - mean) * (gamma / std) + beta
        return self.nonlinearity(normalized)


def batch_norm(layer):
    """
    Convenience function to apply batch normalization to a given layer's output.
    Will steal the layer's nonlinearity if there is one (effectively introducing
    the normalization right before the nonlinearity), and will remove the
    layer's bias if there is one (because it would be redundant).
    @param layer: The `Layer` instance to apply the normalization to; note that
        it will be irreversibly modified as specified above
    @return: A `BatchNormLayer` instance stacked on the given `layer`
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = lasagne.nonlinearities.identity
    if hasattr(layer, 'b'):
        del layer.params[layer.b]
        layer.b = None
    return BatchNormLayer(layer, nonlinearity=nonlinearity)
