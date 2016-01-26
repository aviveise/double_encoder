import theano
from lasagne.layers import Layer
from lasagne.random import get_rng
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams


class TiedGaussianNoiseLayer(Layer):
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
    def __init__(self, incoming, sigma=0.1, noise_layer=None, **kwargs):
        super(TiedGaussianNoiseLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.sigma = sigma
        self._master = noise_layer
        self._noise = None

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.sigma == 0:
            return input
        else:
            if not self._master is None:
                self._noise = self._master._noise

            if self._noise is None:
                self._noise = self._srng.normal(input.shape,
                                                avg=0.0,
                                                std=self.sigma)

            return input + self._noise

