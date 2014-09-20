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
from theano import printing

from numpy.linalg import norm

class Trainer(object):

    @staticmethod
    def train(train_set_x, train_set_y, hyper_parameters, symmetric_double_encoder, params, regularization_methods):


        model = Trainer._build_model(train_set_x,
                                     train_set_y,
                                     hyper_parameters,
                                     symmetric_double_encoder,
                                     params,
                                     regularization_methods)

        #Calculating number of batches
        n_training_batches = train_set_x.get_value(borrow=True).shape[0] / hyper_parameters.batch_size

        #print('------------------------')
        #symmetric_double_encoder[0].print_weights()
        #print('------------------------')

        #The training phase, for each epoch we train on every batch
        for epoch in numpy.arange(hyper_parameters.epochs):
            loss = 0
            for index in xrange(n_training_batches):
                loss += model(index)

            print 'epoch (%d) ,Loss = %f\n' % (epoch, loss / n_training_batches)

        #print('------------------------')
        #symmetric_double_encoder[0].print_weights()
        #print('------------------------')

        del model

    @staticmethod
    def _build_model(train_set_x, train_set_y, hyper_parameters, symmetric_double_encoder, params, regularization_methods):

        #Retrieve the reconstructions of x and y
        x_tilde = symmetric_double_encoder.reconstruct_x()
        y_tilde = symmetric_double_encoder.reconstruct_y()

        var_x = symmetric_double_encoder.var_x
        var_y = symmetric_double_encoder.var_y

        #x_tilde = printing.Print('x_tilde: ')(x_tilde)
        #y_tilde = printing.Print('y_tilde: ')(y_tilde)

        #Index for iterating batches
        index = Tensor.lscalar()

        #Compute the loss of the forward encoding as L2 loss
        loss_backward = ((var_x - x_tilde) ** 2).sum() / hyper_parameters.batch_size

        #Compute the loss of the backward encoding as L2 loss
        loss_forward = ((var_y - y_tilde) ** 2).sum() / hyper_parameters.batch_size

        loss = loss_backward + loss_forward

        #Add the regularization method computations to the loss
        loss += sum([regularization_method.compute(symmetric_double_encoder, params) for regularization_method in regularization_methods])

        #Computing the gradient for the stochastic gradient decent
        #the result is gradients for each parameter of the cross encoder
        gradients = Tensor.grad(loss, params)

        if hyper_parameters.momentum > 0:

            model_updates = [shared(value=numpy.zeros(p.get_value().shape, dtype=config.floatX),
                                                    name='inc_' + p.name) for p in params]

            updates = OrderedDict()
            zipped = zip(params, gradients, model_updates)
            for param, gradient, model_update in zipped:

                delta = hyper_parameters.momentum * model_update - hyper_parameters.learning_rate * gradient

                updates[param] = param + delta
                updates[model_update] = delta

        else:
            #generate the list of updates, each update is a round in the decent
            updates = []
            for param, gradient in zip(params, gradients):
                updates.append((param, param - hyper_parameters.learning_rate * gradient))


        #Building the theano function
        #input : batch index
        #output : both losses
        #updates : gradient decent updates for all params
        #givens : replacing inputs for each iteration
        model = function(inputs=[index],
                         outputs=Out((Tensor.cast(loss, config.floatX)), borrow=True),
                         updates=updates,
                         givens={var_x: train_set_x[index * hyper_parameters.batch_size:
                                                            (index + 1) * hyper_parameters.batch_size, :],
                                 var_y: train_set_y[index * hyper_parameters.batch_size:
                                                            (index + 1) * hyper_parameters.batch_size, :]})

        return model
