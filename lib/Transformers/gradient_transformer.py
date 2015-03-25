__author__ = 'aviv'

import numpy

from theano import function
from theano import tensor as Tensor

from transformer_base import TransformerBase

class GradientTransformer(TransformerBase):

    def __init__(self, double_encoder, params, hyperparameters):
        super(GradientTransformer, self).__init__(double_encoder)

        self._params = params
        self._hyperparameters = hyperparameters

    def compute_outputs(self, set_x, set_y, batch_size):

        gradients = {}
        model = self._build_gradient_model()

        # generate positive samples
        for i in range(set_x.shape[0]):

            raw_sample_gradients = model(set_x[i, :].reshape((1, set_x.shape[1])), set_y[i, :].reshape(1, set_y.shape[1]))
            sample_gradients = {}

            for idx, gradient in enumerate(raw_sample_gradients):
                sample_gradients[self._params[idx].name] = gradient

            gradients[str(i)] = gradient

        return gradients

    def _build_gradient_model(self):

        #Retrieve the reconstructions of x and y
        x_tilde = self._correlation_optimizer.reconstruct_x()
        y_tilde = self._correlation_optimizer.reconstruct_y()

        var_x = self._correlation_optimizer.var_x
        var_y = self._correlation_optimizer.var_y

        #Compute the loss of the forward encoding as L2 loss
        loss_backward = ((var_x - x_tilde) ** 2).sum(dtype=Tensor.config.floatX,
                                                     acc_dtype=Tensor.config.floatX) / self._hyperparameters.batch_size

        #Compute the loss of the backward encoding as L2 loss
        loss_forward = ((var_y - y_tilde) ** 2).sum(dtype=Tensor.config.floatX,
                                                    acc_dtype=Tensor.config.floatX) / self._hyperparameters.batch_size

        loss = loss_backward + loss_forward

        gradients = Tensor.grad(loss, self._params)

        model = function(inputs=[var_x, var_y], outputs=gradients)

        return model