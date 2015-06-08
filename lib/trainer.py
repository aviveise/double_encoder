import cv2
import itertools
from theano.ifelse import IfElse, ifelse
from theano.tensor.nlinalg import matrix_inverse

__author__ = 'aviv'

import sys
import numpy
import theano.tensor as Tensor
import math

from Testers.trace_correlation_tester import TraceCorrelationTester
from Transformers.double_encoder_transformer import DoubleEncoderTransformer

from theano.compat.python2x import OrderedDict
from theano import function, theano
from theano import shared
from theano import printing

from MISC.utils import calculate_reconstruction_error

from numpy.random import RandomState

from MISC.logger import OutputLog


def shuffleDataSet(samplesX, samplesY, random_stream):
    set_size = samplesX.shape[0]
    indices = random_stream.permutation(set_size)

    return samplesX[indices, :], samplesY[indices, :]


def _print_array(x):
    output = ''

    if x.ndim == 0:
        output += '{0:.2f}'.format(float(x))

    elif x.ndim == 1:
        output += '['
        for item in x:
            output += '{0:.2f}'.format(item)
        output += ']'
    else:
        output += '['
        for line in x:
            output += '['
            for item in line:
                output += '{0:.2f}'.format(item)
            output += ']'
        output += ']'

    return output


class Trainer(object):
    @staticmethod
    def train(train_set_x,
              train_set_y,
              hyper_parameters,
              symmetric_double_encoder,
              params,
              regularization_methods,
              print_verbose=False,
              top=0,
              validation_set_x=None,
              validation_set_y=None,
              moving_averages=None):

        # Calculating number of batches
        n_training_batches = int(train_set_x.shape[0] / hyper_parameters.batch_size)
        random_stream = RandomState()

        model_updates = [shared(p.get_value() * 0, borrow=True) for p in params]
        model_deltas = [shared(p.get_value() * 0, borrow=True) for p in params]

        negative_indices = shared(numpy.ones(hyper_parameters.batch_size * 2, dtype=Tensor.config.floatX), borrow=True)

        eps = 1e-8

        symmetric_double_encoder.set_eval(False)

        print 'Building model'
        model = Trainer._build_model(hyper_parameters,
                                     symmetric_double_encoder,
                                     params,
                                     regularization_methods,
                                     model_updates,
                                     model_deltas,
                                     moving_averages,
                                     n_training_batches,
                                     hyper_parameters.training_strategy,
                                     hyper_parameters.rho,
                                     eps)

        # numpy.set_string_function(_print_array, repr=False)

        # The training phase, for each epoch we train on every batch
        best_loss = 0
        for epoch in numpy.arange(hyper_parameters.epochs):

            OutputLog().write('----------Starting Epoch ({0})-----------'.format(epoch), 'debug')

            OutputLog().write('Shuffling dataset', 'debug')
            indices_positive = random_stream.permutation(train_set_x.shape[0])

            loss_forward = 0
            loss_backward = 0

            OutputLog().write('Training {0} batches'.format(n_training_batches), 'debug')
            for index in xrange(n_training_batches):

                start_tick = cv2.getTickCount()

                # need to convert the input into tensor variable
                symmetric_double_encoder.var_x.set_value(
                    train_set_x[indices_positive[index * hyper_parameters.batch_size:
                    (index + 1) * hyper_parameters.batch_size], :], borrow=True)

                symmetric_double_encoder.var_y.set_value(
                    train_set_y[indices_positive[index * hyper_parameters.batch_size:
                    (index + 1) * hyper_parameters.batch_size], :], borrow=True)

                output = model()
                loss_backward += output[0]
                loss_forward += output[1]

                if math.isnan(loss_backward) or math.isnan(loss_forward):
                    OutputLog().write('loss equals NAN, exiting')
                    sys.exit(-1)

                tickFrequency = cv2.getTickFrequency()
                current_time = cv2.getTickCount()

                regularizations = [regularization_method for regularization_method in regularization_methods if
                                   regularization_method.weight > 0]

                zipped = zip(output[6:], regularizations)

                string_output = ' '
                for regularization_output, regularization_method in zipped:
                    string_output += '{0}: {1} '.format(regularization_method.regularization_type,
                                                        regularization_output)

                OutputLog().write(
                    'batch {0}/{1} ended, time: {2:.3f}, loss_x: {3}, loss_y: {4}, loss_h: '
                    '{7:.2f} var_x: {5} var_y: {6} {8}'.
                        format(index,
                               n_training_batches,
                               ((current_time - start_tick) / tickFrequency),
                               output[0],
                               output[1],
                               output[2],
                               output[3],
                               calculate_reconstruction_error(output[4], output[5]),
                               string_output), 'debug')

            loss = (loss_forward + loss_backward) / n_training_batches

            OutputLog().write('Average loss_x: {0} loss_y: {1}'.format(loss_backward / n_training_batches,
                                                                       loss_forward / n_training_batches))

            if best_loss == 0 or loss < best_loss:
                best_loss = loss

            else:
                hyper_parameters.learning_rate *= 0.1

            if print_verbose and not validation_set_y is None and not validation_set_x is None:

                OutputLog().write('----------epoch (%d)----------' % epoch, 'debug')

                symmetric_double_encoder.set_eval(True)

                trace_correlation, var, x, y, layer_id = TraceCorrelationTester(validation_set_x, validation_set_y,
                                                                                top). \
                    test(DoubleEncoderTransformer(symmetric_double_encoder, 0),
                         hyper_parameters)

                symmetric_double_encoder.set_eval(False)

                if math.isnan(var):
                    sys.exit(0)

            OutputLog().write('epoch (%d) ,Loss X = %f, Loss Y = %f\n' % (epoch,
                                                                          loss_backward / n_training_batches,
                                                                          loss_forward / n_training_batches), 'debug')

        del model

    @staticmethod
    def train_output_layer(train_set_x, train_set_y, hyper_parameters, symmetric_double_encoder, params,
                           regularization_methods, top=50):

        model = Trainer._build_model_output(train_set_x,
                                            train_set_y,
                                            hyper_parameters,
                                            symmetric_double_encoder,
                                            params,
                                            regularization_methods,
                                            top)

        # Calculating number of batches
        n_training_batches = train_set_x.get_value(borrow=True).shape[0] / hyper_parameters.batch_size

        # The training phase, for each epoch we train on every batch
        for epoch in numpy.arange(hyper_parameters.epochs):
            loss = 0
            for index in xrange(n_training_batches):
                loss += model(index)

            print 'epoch (%d) ,Loss = %f\n' % (epoch, loss / n_training_batches)

        del model

    @staticmethod
    def _build_model(hyper_parameters,
                     symmetric_double_encoder,
                     params,
                     regularization_methods,
                     model_updates,
                     model_deltas,
                     moving_averages,
                     number_of_batches,
                     strategy='SGDCayley',
                     rho=0.5,
                     eps=1e-8,
                     loss='L2'):

        # Retrieve the reconstructions of x and y
        x_tilde = symmetric_double_encoder.reconstruct_x()
        y_tilde = symmetric_double_encoder.reconstruct_y()

        x_hidden = symmetric_double_encoder[0].output_forward_x
        y_hidden = symmetric_double_encoder[0].output_forward_y

        var_x = symmetric_double_encoder.var_x
        var_y = symmetric_double_encoder.var_y

        print 'Calculating Loss'

        if loss == 'L2':
            # Compute the loss of the forward encoding as L2 loss
            loss_backward = Tensor.mean(((var_x - x_tilde) ** 2).sum(axis=1))

            # Compute the loss of the backward encoding as L2 loss
            loss_forward = Tensor.mean(((var_y - y_tilde) ** 2).sum(axis=1))

            loss = loss_backward + loss_forward

        elif loss == 'cosine':

            mod_x = Tensor.sqrt(Tensor.sum(var_x ** 2, 1) + eps)
            mod_x_tilde = Tensor.sqrt(Tensor.sum(x_tilde ** 2, 1) + eps)
            loss_backward = 1 - Tensor.mean(Tensor.diag(Tensor.dot(var_x, x_tilde.T)) / (mod_x * mod_x_tilde))

            mod_y = Tensor.sqrt(Tensor.sum(var_y ** 2, 1) + eps)
            mod_y_tilde = Tensor.sqrt(Tensor.sum(y_tilde ** 2, 1) + eps)
            loss_forward = 1 - Tensor.mean(Tensor.diag(Tensor.dot(var_y, y_tilde.T)) / (mod_y * mod_y_tilde))

            loss = Tensor.mean(loss_forward * loss_backward.T)

        else:
            raise Exception('Loss not recognized')

        print 'Adding regularization'

        # Add the regularization method computations to the loss
        regularizations = [regularization_method.compute(symmetric_double_encoder, params) for regularization_method in
                           regularization_methods if not regularization_method.weight == 0]

        print 'Regularization number = {0}'.format(len(regularizations))

        if len(regularizations) > 0:
            loss += Tensor.sum(regularizations, dtype=Tensor.config.floatX, acc_dtype=Tensor.config.floatX)

        print 'Calculating gradients'

        # Computing the gradient for the stochastic gradient decent
        # the result is gradients for each parameter of the cross encoder
        gradients = Tensor.grad(loss, params)
        loss_gradients = Tensor.grad(Tensor.mean(loss_backward + loss_forward), params)

        if strategy == 'SGD':

            if hyper_parameters.momentum > 0:

                print 'Adding momentum'

                updates = OrderedDict()
                zipped = zip(params, gradients, model_updates)
                for param, gradient, model_update in zipped:
                    delta = hyper_parameters.momentum * model_update - hyper_parameters.learning_rate * gradient

                    updates[param] = param + delta
                    updates[model_update] = delta

            else:
                # generate the list of updates, each update is a round in the decent
                updates = []
                for param, gradient in zip(params, gradients):
                    updates.append((param, param - hyper_parameters.learning_rate * gradient))

        elif strategy == 'adaGrad':
            updates = OrderedDict()
            zipped = zip(params, gradients, model_updates)
            for ndx, (param, gradient, accumulated_gradient) in enumerate(zipped):
                agrad = accumulated_gradient + gradient ** 2
                effective_learning_rate = (hyper_parameters.learning_rate / Tensor.sqrt(agrad + eps))
                update, delta = Trainer._calc_update(effective_learning_rate, gradient, param)
                #delta = effective_learning_rate * gradient
                updates[param] = update
                updates[accumulated_gradient] = agrad

        elif strategy == 'adaDelta':
            updates = OrderedDict()
            zipped = zip(params, gradients, model_updates, model_deltas)
            for ndx, (param, gradient, accumulated_gradient, accumulated_delta) in enumerate(zipped):
                agrad = rho * accumulated_gradient + (1 - rho) * gradient ** 2
                #delta = Tensor.sqrt((accumulated_delta + eps) / (agrad + eps)) * gradient
                step_size = Tensor.sqrt((accumulated_delta + eps) / (agrad + eps))
                update, delta = Trainer._calc_update(step_size, gradient, param)
                updates[param] = update
                updates[accumulated_gradient] = agrad
                updates[accumulated_delta] = rho * accumulated_delta + (1 - rho) * (delta ** 2)

        elif strategy == 'SGDCayley':
            updates = []
            for param, gradient in zip(params, gradients):

                if param.name == 'Wx_layer0' or param.name == 'Wy_layer0':
                    param_update = Trainer._calc_update(hyper_parameters.learning_rate, gradient, param, 'Cayley')
                else:
                    param_update = Trainer._calc_update(hyper_parameters.learning_rate, gradient, param, 'Regular')

                updates.append((param, param_update))

        else:
            msg = 'Unknown optimization strategy'
            OutputLog().write(msg)
            raise Exception(msg)

        print 'Building function'

        variance_hidden_x = Tensor.var(x_hidden, axis=0)
        variance_hidden_y = Tensor.var(y_hidden, axis=0)

        if moving_averages is not None:
            Trainer._add_moving_averages(moving_averages, updates, number_of_batches)

        # Building the theano function
        # input : batch index
        # output : both losses
        # updates : gradient decent updates for all params
        # givens : replacing inputs for each iteration
        model = function(inputs=[],
                         outputs=[Tensor.mean(loss_backward),
                                  Tensor.mean(loss_forward),
                                  Tensor.sum(variance_hidden_x),
                                  Tensor.sum(variance_hidden_y),
                                  x_hidden,
                                  y_hidden] + regularizations,
                         updates=updates)

        return model

    @staticmethod
    def _add_moving_averages(moving_averages, updates, length):

        params = list(itertools.chain(*[i[1] for i in moving_averages]))
        values = list(itertools.chain(*[i[0] for i in moving_averages]))

        factor = 1.0 / length
        for tensor, param in zip(values, params):
            updates[param] = (1.0 - factor) * param + factor * tensor
        return updates

    @staticmethod
    def _calc_update(step_size, gradient, param, type='Cayley'):

        if type == 'Cayley' and (param.name == 'Wx_layer0' or param.name == 'Wy_layer0'):
            A = Tensor.dot((step_size / 2) * gradient, param.T) - Tensor.dot(param, ((step_size / 2) * gradient).T)
            I = Tensor.identity_like(A)
            Q = Tensor.dot(matrix_inverse(I + A), (I - A))
            update = Tensor.dot(Q, param)
            delta = (step_size / 2) * Tensor.dot(A, (param + update))
            return update, delta
        else:
            delta = step_size * gradient
            return param - delta, delta