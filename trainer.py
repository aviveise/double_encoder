__author__ = 'aviv'

import sys
import os

import numpy
import theano.tensor as Tensor

from theano.compat.python2x import OrderedDict
from theano import function
from theano import config
from theano import shared
from theano import Out


class Trainer(object):
    def __init__(self, var_x, var_y, train_set_x, train_set_y, hyper_parameters, regularization_methods):

        self._variable_x = var_x
        self._variable_y = var_y
        self._hyper_parameters = hyper_parameters
        self._train_set_x = train_set_x
        self._train_set_y = train_set_y
        self._regularization_methods = regularization_methods

    def train(self, symmetric_double_encoder, params):

        model = self._build_model(symmetric_double_encoder, params)

        #Calculating number of batches
        n_training_batches = self._train_set_x.get_value(borrow=True).shape[0] / self._hyper_parameters.batch_size

        print 'Starting Training:'

        #The training phase, for each epoch we train on every batch
        for epoch in numpy.arange(self._hyper_parameters.epochs):
            for index in xrange(n_training_batches):
                loss = model(index)

    def _build_model(self, symmetric_double_encoder, params):

        #Retrieve the reconstructions of x and y
        x_tilde = symmetric_double_encoder.reconstruct_x()
        y_tilde = symmetric_double_encoder.reconstruct_y()

        #Index for iterating batches
        index = Tensor.lscalar()

        #Compute the loss of the forward encoding as L2 loss
        loss_backward = ((self._variable_x - x_tilde) ** 2).sum(axis=1).sum() / self._hyper_parameters.batch_size

        #Compute the loss of the backward encoding as L2 loss
        loss_forward = ((self._variable_y - y_tilde) ** 2).sum(axis=1).sum() / self._hyper_parameters.batch_size

        loss = loss_backward + loss_forward

        #Add the regularization method computations to the loss
        loss += sum([regularization_method.compute(symmetric_double_encoder) for regularization_method in self._regularization_methods])

        #Computing the gradient for the stochastic gradient decent
        #the result is gradients for each parameter of the cross encoder
        gradients = Tensor.grad(loss, params)

        if self.hyper_parameters.momentum > 0:

            model_updates = OrderedDict([(p, shared(value=numpy.zeros(p.get_value().shape,
                                                                      dtype=config.floatX),
                                                    name='inc_' + p.name)) for p in params])

            updates = OrderedDict()
            for param, gradient, model_update in zip(params, gradients, model_updates):
                delta = self.hyper_parameters.momentum * model_update - self.hyper_parameters.learning_rate * gradient
                updates[param] = param + delta
                updates[model_update] = delta

        else:
            #generate the list of updates, each update is a round in the decent
            updates = []
            for param, gradient in zip(params, gradients):
                updates.append((param, param - self.hyper_parameters.learning_rate * gradient))


        #Building the theano function
        #input : batch index
        #output : both losses
        #updates : gradient decent updates for all params
        #givens : replacing inputs for each iteration
        model = function(inputs=[index],
                         outputs=Out((Tensor.cast(loss, config.floatX)), borrow=True),
                         updates=updates,
                         givens={self.x1: self._train_set_x[index * self.hyper_parameters.batch_size:
                         (index + 1) * self.hyper_parameters.batch_size, :],
                                 self.x2: self._train_set_y[index * self.hyper_parameters.batch_size:
                                 (index + 1) * self.hyper_parameters.batch_size, :]})

        return model
