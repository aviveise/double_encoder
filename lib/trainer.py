import cv2
import itertools
from theano.ifelse import IfElse, ifelse
from theano.tensor.nlinalg import matrix_inverse
from theano.tensor.shared_randomstreams import RandomStreams
from lib.MISC.theano_ops import batched_inv

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
              moving_averages=None,
              decay=False,
              reduce_val=0):

        OutputLog().write('Using Decay = {0}'.format(decay))

        # Calculating number of batches
        n_training_batches = int(train_set_x.shape[0] / hyper_parameters.batch_size)
        random_stream = RandomState()

        early_stop_count = 0

        model_updates = [shared(p.get_value() * 0) for p in params]
        model_deltas = [shared(p.get_value() * 0) for p in params]

        eps = 1e-8

        symmetric_double_encoder.set_eval(False)

        last_correlation = 0

        correlations = []

        tester = TraceCorrelationTester(validation_set_x, validation_set_y, top)

        learning_rate = hyper_parameters.learning_rate

        # The training phase, for each epoch we train on every batch
        best_loss = 0
        for epoch in numpy.arange(hyper_parameters.epochs):

            OutputLog().write('----------Starting Epoch ({0})-----------'.format(epoch), 'debug')

            print 'Building model'
            model = Trainer._build_model(hyper_parameters,
                                         learning_rate,
                                         symmetric_double_encoder,
                                         params,
                                         regularization_methods,
                                         model_updates,
                                         model_deltas,
                                         moving_averages,
                                         n_training_batches,
                                         hyper_parameters.training_strategy,
                                         0.9,
                                         0.999,
                                         hyper_parameters.rho,
                                         eps,
                                         'L2',
                                         len(symmetric_double_encoder) - 1)

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

                output = model(index + 1)
                loss_backward += output[0]
                loss_forward += output[1]

                if math.isnan(loss_backward) or math.isnan(loss_forward):
                    OutputLog().write('loss equals NAN, exiting')
                    sys.exit(-1)

                tickFrequency = cv2.getTickFrequency()
                current_time = cv2.getTickCount()

                regularizations = [regularization_method for regularization_method in regularization_methods if not
                regularization_method.weight == 0]

                string_output = ''

                if len(regularizations) > 0:
                    zipped = zip(output[8:8 + len(regularizations)], regularizations)

                    string_output = ' '
                    for regularization_output, regularization_method in zipped:
                        string_output += '{0}: {1} '.format(regularization_method.regularization_type,
                                                            regularization_output)

                OutputLog().write(
                    'batch {0}/{1} ended, time: {2:.3f}, loss_x: {3}, loss_y: {4}, loss_h: '
                    '{7:.2f} var_x: {5} var_y: {6} mean_g: {9} var_g: {10} {8}'.
                        format(index,
                               n_training_batches,
                               ((current_time - start_tick) / tickFrequency),
                               output[0],
                               output[1],
                               output[2],
                               output[3],
                               calculate_reconstruction_error(output[4], output[5]),
                               string_output,
                               numpy.mean(output[6]),
                               numpy.mean(output[7])), 'debug')

            OutputLog().write('Average loss_x: {0} loss_y: {1}'.format(loss_backward / (n_training_batches * 2),
                                                                       loss_forward / (n_training_batches * 2)))

            correlations = [-1]

            if print_verbose and not validation_set_y is None and not validation_set_x is None:

                OutputLog().write('----------epoch (%d)----------' % epoch, 'debug')

                symmetric_double_encoder.set_eval(True)

                correlations, best_correlation, var, x, y, layer_id = tester.test(
                    DoubleEncoderTransformer(symmetric_double_encoder, 0),
                    hyper_parameters)

                symmetric_double_encoder.set_eval(False)

                if math.isnan(var):
                    sys.exit(0)

            if last_correlation == max(correlations):
                early_stop_count += 1

            if hyper_parameters.decay_factor > 0:
                if not hyper_parameters.decay:
                    if last_correlation == 0:
                        last_correlation = max(correlations)
                    else:
                        if last_correlation - max(correlations) > 0.1:
                            OutputLog().write('Decaying learning rate')
                            learning_rate *= hyper_parameters.decay_factor

                        last_correlation = max(correlations)
                else:
                    if epoch in hyper_parameters.decay:
                        OutputLog().write('Decaying learning rate')
                        learning_rate *= hyper_parameters.decay_factor
                        symmetric_double_encoder.export_encoder(OutputLog().output_path, 'epoch_{0}'.format(epoch))
            else:
                if last_correlation == 0:
                    last_correlation = max(correlations)
                elif abs(last_correlation - max(correlations)) < 0.5:
                    break

            if early_stop_count == 3:
                break

            OutputLog().write('epoch (%d) ,Loss X = %f, Loss Y = %f, learning_rate = %f\n' % (epoch,
                                                                                              loss_backward / n_training_batches,
                                                                                              loss_forward / n_training_batches,
                                                                                              learning_rate
                                                                                              ), 'debug')

        tester.saveResults(OutputLog().output_path)

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
                     learning_rate,
                     symmetric_double_encoder,
                     params,
                     regularization_methods,
                     model_updates,
                     model_deltas,
                     moving_averages,
                     number_of_batches,
                     strategy='SGDCayley',
                     bias_1=0.9,
                     bias_2=0.999,
                     rho=0.5,
                     eps=1e-8,
                     loss='L2',
                     last_layer=0):

        # loss_decision = Tensor.iscalar()
        t = Tensor.dscalar()

        # Retrieve the reconstructions of x and y
        x_tilde = symmetric_double_encoder.reconstruct_x()
        y_tilde = symmetric_double_encoder.reconstruct_y()

        x_hidden = symmetric_double_encoder[0].output_forward_x
        y_hidden = symmetric_double_encoder[0].output_forward_y

        var_x = symmetric_double_encoder.var_x
        var_y = symmetric_double_encoder.var_y

        print 'Calculating Loss'

        if loss == 'L2':
            loss_backward = Tensor.mean(
                ((var_x - x_tilde) ** 2).sum(axis=1, dtype=Tensor.config.floatX))

            # Compute the loss of the backward encoding as L2 loss
            loss_forward = Tensor.mean(
                ((var_y - y_tilde) ** 2).sum(axis=1, dtype=Tensor.config.floatX))

            # loss = ifelse(loss_decision, loss_forward, loss_backward)#loss_backward + loss_forward
            loss = (loss_forward + loss_backward)

        elif loss == 'cosine':

            mod_x = Tensor.sqrt(Tensor.sum(var_x ** 2, 1) + eps)
            mod_x_tilde = Tensor.sqrt(Tensor.sum(x_tilde ** 2, 1) + eps)
            loss_backward = 1 - Tensor.mean(Tensor.diag(Tensor.dot(var_x, x_tilde.T)) / (mod_x * mod_x_tilde))

            mod_y = Tensor.sqrt(Tensor.sum(var_y ** 2, 1) + eps)
            mod_y_tilde = Tensor.sqrt(Tensor.sum(y_tilde ** 2, 1) + eps)
            loss_forward = 1 - Tensor.mean(Tensor.diag(Tensor.dot(var_y, y_tilde.T)) / (mod_y * mod_y_tilde))

            loss = Tensor.mean(loss_forward * loss_backward.T)

            # Compute the loss of the forward encoding as L2 loss

        else:
            raise Exception('Loss not recognized')

        loss -= Trainer.add_negative(var_x, x_tilde, None)
        loss -= Trainer.add_negative(var_y, y_tilde, None)

        print 'Adding regularization'

        # Add the regularization method computations to the loss
        regularizations = [regularization_method.compute(symmetric_double_encoder, params) for regularization_method in
                           regularization_methods if not regularization_method.weight == 0]

        print 'Regularization number = {0}'.format(len(regularizations))

        if len(regularizations) > 0:
            loss += Tensor.sum(regularizations, dtype=Tensor.config.floatX)

        print 'Calculating gradients'

        # Computing the gradient for the stochastic gradient decent
        # the result is gradients for each parameter of the cross encoder
        gradients = Tensor.grad(loss, params)

        if strategy == 'SGD':

            if hyper_parameters.momentum > 0:

                print 'Adding momentum'

                updates = OrderedDict()
                zipped = zip(params, gradients, model_updates)
                for param, gradient, model_update in zipped:
                    update, delta = Trainer._calc_update(learning_rate, gradient, param,
                                                         last_layer=last_layer)
                    delta = hyper_parameters.momentum * model_update - delta

                    updates[param] = param + delta
                    updates[model_update] = delta

            else:
                # generate the list of updates, each update is a round in the decent
                updates = OrderedDict()
                for param, gradient in zip(params, gradients):
                    update, delta = Trainer._calc_update(learning_rate, gradient, param,
                                                         last_layer=last_layer)
                    updates[param] = update

        elif strategy == 'adaGrad':
            updates = OrderedDict()
            zipped = zip(params, gradients, model_updates)
            for ndx, (param, gradient, accumulated_gradient) in enumerate(zipped):
                agrad = accumulated_gradient + gradient ** 2
                effective_learning_rate = (learning_rate / (Tensor.sqrt(agrad) + eps))
                update, delta = Trainer._calc_update(effective_learning_rate, gradient, param, last_layer=last_layer)
                # delta = effective_learning_rate * gradient
                updates[param] = update
                updates[accumulated_gradient] = agrad

        elif strategy == 'RMSProp':
            updates = OrderedDict()
            zipped = zip(params, gradients, model_updates)
            for ndx, (param, gradient, accumulated_gradient) in enumerate(zipped):
                agrad = rho * accumulated_gradient + (1 - rho) * gradient ** 2
                effective_learning_rate = (learning_rate / (Tensor.sqrt(agrad) + eps))
                update, delta = Trainer._calc_update(effective_learning_rate, gradient, param, last_layer=last_layer)
                # delta = effective_learning_rate * gradient
                updates[param] = update
                updates[accumulated_gradient] = agrad

        elif strategy == 'RMSProp2':
            updates = OrderedDict()
            zipped = zip(params, gradients, model_updates)
            for ndx, (param, gradient, accumulated_gradient) in enumerate(zipped):
                agrad = rho * accumulated_gradient + gradient ** 2
                effective_learning_rate = (learning_rate / (Tensor.sqrt(agrad) + eps))
                update, delta = Trainer._calc_update(effective_learning_rate, gradient, param, last_layer=last_layer)
                # delta = effective_learning_rate * gradient
                updates[param] = update
                updates[accumulated_gradient] = agrad

        elif strategy == 'adaDelta':
            updates = OrderedDict()
            zipped = zip(params, gradients, model_updates, model_deltas)
            for ndx, (param, gradient, accumulated_gradient, accumulated_delta) in enumerate(zipped):
                agrad = rho * accumulated_gradient + (1 - rho) * gradient ** 2
                # delta = Tensor.sqrt((accumulated_delta + eps) / (agrad + eps)) * gradient
                step_size = Tensor.sqrt((accumulated_delta + eps) / (agrad + eps))
                update, delta = Trainer._calc_update(step_size, gradient, param, last_layer=last_layer)
                updates[param] = update
                updates[accumulated_gradient] = agrad
                updates[accumulated_delta] = rho * accumulated_delta + (1 - rho) * (delta ** 2)

        elif strategy == 'adam':
            updates = OrderedDict()
            zipped = zip(params, gradients, model_updates, model_deltas)
            for ndx, (param, gradient, accumulated_gradient, accumulated_delta) in enumerate(zipped):
                moment_1 = bias_1 * accumulated_gradient + (1 - bias_1) * gradient
                moment_2 = bias_2 * accumulated_delta + (1 - bias_2) * gradient ** 2
                corrected_moment_1 = moment_1 / Tensor.cast((1 - bias_1 ** t), theano.config.floatX)
                corrected_moment_2 = moment_2 / Tensor.cast((1 - bias_2 ** t), theano.config.floatX)
                g = corrected_moment_1 / (Tensor.sqrt(corrected_moment_2 + eps))
                update, delta = Trainer._calc_update(learning_rate, g, param, last_layer=last_layer)

                updates[param] = update
                updates[accumulated_gradient] = moment_1
                updates[accumulated_delta] = moment_2
        elif strategy == 'SGDCayley':
            updates = []
            for param, gradient in zip(params, gradients):

                if param.name == 'Wx_layer0' or param.name == 'Wy_layer0':
                    param_update = Trainer._calc_update(learning_rate, gradient, param, 'Cayley',
                                                        last_layer=last_layer)
                else:
                    param_update = Trainer._calc_update(learning_rate, gradient, param, 'Regular',
                                                        last_layer=last_layer)

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

        update_mean = []
        update_var = []
        for param in params:
            if 'W' in param.name:
                OutputLog().write('Adding weight: {0}'.format(param.name))
                update_mean.append(Tensor.mean(abs(updates[param])))
                update_var.append(Tensor.var(abs(updates[param])))

        update_mean = Tensor.stacklists(update_mean)
        update_var = Tensor.stacklists(update_var)

        # Building the theano function
        # input : batch index
        # output : both losses
        # updates : gradient decent updates for all params
        # givens : replacing inputs for each iteration

        model = function(inputs=[t],
                         outputs=[Tensor.mean(loss_backward),
                                  Tensor.mean(loss_forward),
                                  Tensor.mean(variance_hidden_x),
                                  Tensor.mean(variance_hidden_y),
                                  x_hidden,
                                  y_hidden, update_mean, update_var] + regularizations + [t],
                         updates=updates)

        return model

    @staticmethod
    def _add_moving_averages(moving_averages, updates, length, factor=0.1):

        params = list(itertools.chain(*[i[1] for i in moving_averages]))
        values = list(itertools.chain(*[i[0] for i in moving_averages]))

        if not factor:
            factor = 1.0 / length
        for tensor, param in zip(values, params):
            updates[param] = (1.0 - factor) * param + factor * tensor
        return updates

    @staticmethod
    def _calc_update(step_size, gradient, param, type='Cayley', last_layer=0, hidden_x=None, hidden_y=None):

        # 'W' in param.name:
        if type == 'Cayley' and (param.name == 'Wx_layer0' or param.name == 'Wy_layer{0}'.format(last_layer)):
            OutputLog().write('Adding constraint to {0}:'.format(param.name))
            A = Tensor.dot(((step_size / 2) * gradient).T, param) - Tensor.dot(param.T, ((step_size / 2) * gradient))
            I = Tensor.identity_like(A)
            temp = I + A
            Q = Tensor.dot(batched_inv(temp.dimshuffle('x',0,1)).reshape(temp.shape, ndim=2), (I - A))
            #Q = Tensor.dot(matrix_inverse(temp), I-A)
            update = Tensor.dot(param, Q)
            delta = (step_size / 2) * Tensor.dot((param + update), A)
            return update, delta
        else:
            delta = step_size * gradient
            return param - delta, delta

    @staticmethod
    def get_output(symmetric_double_encoder, layer):
        x_hidden = symmetric_double_encoder[layer].output_forward_x
        y_hidden = symmetric_double_encoder[layer].output_forward_y

        model = function([], [x_hidden, y_hidden])
        return model()

    @classmethod
    def add_negative(cls, var_x, x_tilde, type='samples'):

        if type is None:
            return 0

        random_stream = RandomStreams()
        if type == 'samples':
            n = var_x.shape[0]
            perm = random_stream.permutation(n=n)
            shuffled_var_x = var_x[perm, :]
            return Tensor.mean(((shuffled_var_x - x_tilde) ** 2).sum(axis=1))

        if type == 'features':
            n = var_x.shape[1]
            perm = random_stream.permutation(n=n)
            shuffled_var_x = var_x[:, perm]
            return Tensor.mean(((shuffled_var_x - x_tilde) ** 2).sum(axis=1))
