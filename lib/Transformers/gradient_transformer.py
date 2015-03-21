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

        gradients = []
        model = self._build_gradient_model()
        number_of_batches = set_x.shape[0] / batch_size

        # generate positive samples
        for i in range(set_x.shape[0]):
            gradients.append(model(set_x[i, :].reshape((1, set_x.shape[1])), set_y[i, :].reshape(1, set_y.shape[1])))

        ## generate positive samples
        #for i in range(number_of_batches):

        #    j = i
        #    while j == i:
        #        j = numpy.random.uniform(high=number_of_batches, size=1)

        #    length_x = min((i + 1) * batch_size, set_x.shape[0])
        #    length_y = min((j + 1) * batch_size, set_x.shape[0])

        #    permutation = numpy.random.permutation(numpy.arange(j * batch_size, length_y))

        #    gradients.append(model(set_x[i * batch_size: length_x, :], set_y[permutation, :]))


        # generate positive samples
        #for i in range(number_of_batches):

        #    j = i
        #    while j == i:
        #        j = numpy.random.uniform(high=number_of_batches, size=1)

        #    length_x = min((i + 1) * batch_size, set_x.shape[0])
        #    length_y = min((j + 1) * batch_size, set_x.shape[0])

        #    permutation = numpy.random.permutation(numpy.arange(j * batch_size, length_y))

        #    gradients.append(model(set_x[i * batch_size: length_x, :], set_y[permutation, :]))

        samples = []
        for index, gradient_vector in enumerate(gradients):

            for grad_idx, gradient in enumerate(gradient_vector):

                print gradient
                print grad_idx
                if grad_idx == 0:
                    sample = numpy.array(gradient).reshape(-1)
                else:
                    sample = numpy.concatenate(sample, numpy.array(gradient).reshape(-1))

            samples.append(sample)


        results = numpy.ndarray(len(samples), samples[0].shape[0])

        for idx, sample in enumerate(samples):
            results[idx, :] = sample

        return results

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