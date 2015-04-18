import cv2
from MISC.logger import OutputLog

__author__ = 'aviv'

import sys
import numpy
import theano.tensor as Tensor
import theano.tensor.nlinalg
import theano.tensor.slinalg
import theano.printing
import math

from Testers.trace_correlation_tester import TraceCorrelationTester
from Transformers.double_encoder_transformer import DoubleEncoderTransformer

from theano.compat.python2x import OrderedDict
from theano import function
from theano import config
from theano import shared
from theano import Out
from numpy.random import RandomState

def shuffleDataSet(samplesX, samplesY, random_stream):

    set_size = samplesX.shape[0]
    indices = random_stream.permutation(set_size)

    return samplesX[indices, :], samplesY[indices, :]


class Trainer(object):

    @staticmethod
    def train(train_set_x,
              train_set_y,
              hyper_parameters,
              symmetric_double_encoder,
              params,
              regularization_methods,
              print_verbose=False,
              top=50,
              validation_set_x=None,
              validation_set_y=None):



        #Calculating number of batches
        n_training_batches = int(train_set_x.shape[0] / hyper_parameters.batch_size)

        random_stream = RandomState()

        #The training phase, for each epoch we train on every batch
        for epoch in numpy.arange(hyper_parameters.epochs):

            print '----------Starting Epoch ({0})-----------'.format(epoch)

            print 'Shuffling dataset'
            #train_set_x, train_set_y = shuffleDataSet(train_set_x, train_set_y, random_stream)
            indices = random_stream.permutation(train_set_x.shape[0])

            print 'Building model'
            model = Trainer._build_model(hyper_parameters,
                                         symmetric_double_encoder,
                                         params,
                                         regularization_methods)

            loss_forward = 0
            loss_backward = 0

            print 'Training {0} batches'.format(n_training_batches)
            for index in xrange(n_training_batches):

                start_tick = cv2.getTickCount()

                #need to convert the input into tensor variable
                symmetric_double_encoder.var_x.set_value(train_set_x[indices[index * hyper_parameters.batch_size:
                                                            (index + 1) * hyper_parameters.batch_size], :], borrow=True)
                symmetric_double_encoder.var_y.set_value(train_set_y[indices[index * hyper_parameters.batch_size:
                                                            (index + 1) * hyper_parameters.batch_size], :], borrow=True)

                output = model()
                loss_forward += output[0]
                loss_backward += output[1]

                tickFrequency = cv2.getTickFrequency()
                current_time = cv2.getTickCount()

                OutputLog().write('batch {0}/{1} ended, time: {2}'.format(index,
                                                                          n_training_batches,
                                                                          ((current_time - start_tick) / tickFrequency)))

            if print_verbose and not validation_set_y is None and not validation_set_x is None:

                print '----------epoch (%d)----------\n' % epoch

                trace_correlation, var, x, y, layer_id = TraceCorrelationTester(validation_set_x.T, validation_set_y.T, top).\
                    test(DoubleEncoderTransformer(symmetric_double_encoder, 0),
                         hyper_parameters)

                if var < 1e-4:
                    return

            else:
                print 'epoch (%d) ,Loss X = %f, Loss Y = %f\n' % (epoch,
                                                                  loss_backward / n_training_batches,
                                                                  loss_forward / n_training_batches)

        del model

    @staticmethod
    def train_output_layer(train_set_x, train_set_y, hyper_parameters, symmetric_double_encoder, params, regularization_methods, top=50):

        model = Trainer._build_model_output(train_set_x,
                                            train_set_y,
                                            hyper_parameters,
                                            symmetric_double_encoder,
                                            params,
                                            regularization_methods,
                                            top)

        #Calculating number of batches
        n_training_batches = train_set_x.get_value(borrow=True).shape[0] / hyper_parameters.batch_size

        #The training phase, for each epoch we train on every batch
        for epoch in numpy.arange(hyper_parameters.epochs):
            loss = 0
            for index in xrange(n_training_batches):
                loss += model(index)

            print 'epoch (%d) ,Loss = %f\n' % (epoch, loss / n_training_batches)

        del model

    @staticmethod
    def _build_model(hyper_parameters, symmetric_double_encoder, params, regularization_methods):

        #Retrieve the reconstructions of x and y
        x_tilde = symmetric_double_encoder.reconstruct_x()
        y_tilde = symmetric_double_encoder.reconstruct_y()

        var_x = symmetric_double_encoder.var_x
        var_y = symmetric_double_encoder.var_y

        #Index for iterating batches
        index = Tensor.lscalar()

        print 'Calculating Loss'

        #Compute the loss of the forward encoding as L2 loss
        loss_backward = ((var_x - x_tilde) ** 2).sum(dtype=Tensor.config.floatX,
                                                     acc_dtype=Tensor.config.floatX) / hyper_parameters.batch_size

        #Compute the loss of the backward encoding as L2 loss
        loss_forward = ((var_y - y_tilde) ** 2).sum(dtype=Tensor.config.floatX,
                                                    acc_dtype=Tensor.config.floatX) / hyper_parameters.batch_size

        loss = loss_backward + loss_forward

        print 'Adding regularization'

        #Add the regularization method computations to the loss
        regularizations = [regularization_method.compute(symmetric_double_encoder, params) for regularization_method in regularization_methods if regularization_method.weight > 0]

        print 'Regularization number = {0}'.format(len(regularizations))

        if len(regularizations) > 0:
            loss += Tensor.sum(regularizations, dtype=Tensor.config.floatX, acc_dtype=Tensor.config.floatX)

        print 'Calculating gradients'

        #Computing the gradient for the stochastic gradient decent
        #the result is gradients for each parameter of the cross encoder
        gradients = Tensor.grad(loss, params)

        if hyper_parameters.momentum > 0:

            print 'Adding momentum'

            model_updates = [shared(p.get_value() * 0, borrow=True) for p in params]

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

        print 'Building function'

        #Building the theano function
        #input : batch index
        #output : both losses
        #updates : gradient decent updates for all params
        #givens : replacing inputs for each iteration
        model = function(inputs=[],
                         outputs=[loss_backward, loss_forward],
                         updates=updates)

        return model

    @staticmethod
    def _build_model_output(train_set_x, train_set_y, hyper_parameters, symmetric_double_encoder, params, regularization_methods, top):

        index = Tensor.scalar()

        var_x = symmetric_double_encoder.var_x
        var_y = symmetric_double_encoder.var_y

        output_x = symmetric_double_encoder.output_x.T
        output_y = symmetric_double_encoder.output_y.T

        batch_size = output_x.shape[1]
        output_size = output_x.shape[0]

        reg1 = hyper_parameters.reg1
        reg2 = hyper_parameters.reg2

        ones = Tensor.ones([batch_size, batch_size], dtype=Tensor.config.floatX)

        centered_x = output_x - Tensor.dot(output_x, ones) / batch_size
        centered_y = output_y - Tensor.dot(output_y, ones) / batch_size

        s12 = Tensor.nlinalg.diag(Tensor.nlinalg.diag(Tensor.dot(centered_x, centered_y.T))) / (batch_size - 1)
        s11 = Tensor.nlinalg.diag(Tensor.nlinalg.diag(Tensor.dot(centered_x, centered_x.T))) / (batch_size - 1) + reg1 * Tensor.eye(output_size, output_size, dtype=Tensor.config.floatX)
        s22 = Tensor.nlinalg.diag(Tensor.nlinalg.diag(Tensor.dot(centered_y, centered_y.T))) / (batch_size - 1) + reg2 * Tensor.eye(output_size, output_size, dtype=Tensor.config.floatX)

        s11_chol = Tensor.slinalg.cholesky(s11)
        s22_chol = Tensor.slinalg.cholesky(s22)

        s11_inv = Tensor.nlinalg.matrix_inverse(s11_chol)
        s22_inv = Tensor.nlinalg.matrix_inverse(s22_chol)

        T_mat= Tensor.dot(Tensor.dot(s11_inv, s12), s22_inv.T)

        loss = -(Tensor.nlinalg.trace(Tensor.slinalg.cholesky(Tensor.dot(T_mat.T, T_mat)))) #+ 10 ** (-2) * Tensor.eye(output_size, output_size))))

        #loss += sum([regularization_method.compute(symmetric_double_encoder, params) for regularization_method in regularization_methods])

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
