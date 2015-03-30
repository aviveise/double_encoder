import scipy


__author__ = 'aviv'

import numpy

from theano import function
from theano import tensor as Tensor
from MISC.logger import OutputLog
from transformer_base import TransformerBase

def lincompress(x):
    U, S, V = scipy.linalg.svd(numpy.dot(x.T, x))
    xc = numpy.dot(U, numpy.sqrt(S)).T
    return xc



class GradientTransformer(TransformerBase):

    def __init__(self, double_encoder, params, hyperparameters):
        super(GradientTransformer, self).__init__(double_encoder)

        self._params = params
        self._hyperparameters = hyperparameters

    def compute_outputs(self, set_x, set_y, batch_size):

        model = self._build_gradient_model()

        # generate positive samples
        for i in range(set_x.shape[0]):

            sample_gradients = model(set_x[i, :].reshape((1, set_x.shape[1])), set_y[i, :].reshape(1, set_y.shape[1]))

            if i == 0:
                gradients = sample_gradients.reshape(1, sample_gradients.shape[0])
            else:
                gradients = numpy.concatenate(gradients, sample_gradients)

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

        gradients = Tensor.concatenate(gradients)

        model = function(inputs=[var_x, var_y], outputs=gradients)

        return model