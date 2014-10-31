__author__ = 'aviv'

import numpy
from theano import tensor as Tensor
from theano import scan
from theano.tensor import nlinalg
from theano.compat.python2x import OrderedDict
from theano import function
from theano import config
from theano import shared
from theano import Out

from training_strategy import TrainingStrategy
from stacked_double_encoder import StackedDoubleEncoder
from Layers.symmetric_hidden_layer import SymmetricHiddenLayer
from MISC.logger import OutputLog
from trainer import Trainer

class IterativeTrainingStrategy(TrainingStrategy):

    def __init__(self):
        super(IterativeTrainingStrategy, self).__init__()
        OutputLog().write('\nStrategy: Iterative')
        self.name = 'iterative'

    def train(self,
              training_set_x,
              training_set_y,
              hyper_parameters,
              regularization_methods,
              activation_method):

        #need to convert the input into tensor variable
        training_set_x = shared(training_set_x, 'training_set_x', borrow=True)
        training_set_y = shared(training_set_y, 'training_set_y', borrow=True)

        symmetric_double_encoder = StackedDoubleEncoder(hidden_layers=[],
                                                        numpy_range=self._random_range,
                                                        input_size=training_set_x.get_value(borrow=True).shape[1],
                                                        output_size=training_set_y.get_value(borrow=True).shape[1],
                                                        activation_method=activation_method)

        #In this phase we train the stacked encoder one layer at a time
        #once a layer was added, weights not belonging to the new layer are
        #not changed
        for layer_size in hyper_parameters.layer_sizes:

            print '--------Adding Layer of Size - %d--------\n' % layer_size
            self._add_cross_encoder_layer(layer_size,
                                          symmetric_double_encoder,
                                          hyper_parameters.method_in,
                                          hyper_parameters.method_out)

            params = []
            params.extend(symmetric_double_encoder[-1].x_params)
            params.extend(symmetric_double_encoder[-1].y_params)

            Trainer.train(train_set_x=training_set_x,
                          train_set_y=training_set_y,
                          hyper_parameters=hyper_parameters,
                          symmetric_double_encoder=symmetric_double_encoder,
                          params=params,
                          regularization_methods=regularization_methods)

        #self._train_hidden_layer(training_set_x, training_set_y, 0, symmetric_double_encoder, hyper_parameters)

        return symmetric_double_encoder

    def _train_hidden_layer(self, training_x, training_y, layer_num, symmetric_double_encoder, hyper_parameters):

        #Index for iterating batches
        index = Tensor.lscalar()

        layer = symmetric_double_encoder[layer_num]

        params = []
        params.extend(layer.x_hidden_params)
        params.extend(layer.y_hidden_params)

        var_x = symmetric_double_encoder.var_x
        var_y = symmetric_double_encoder.var_y

        forward = layer.output_forward
        backward = layer.output_backward

        forward_centered = (forward - Tensor.mean(forward, axis=0)).T
        backward_centered = (backward - Tensor.mean(backward, axis=0)).T

        forward_var = Tensor.dot(forward_centered, forward_centered.T) + 0.1 * Tensor.eye(forward_centered.shape[0])
        backward_var = Tensor.dot(backward_centered, backward_centered.T) + 0.1 * Tensor.eye(backward_centered.shape[0])

        e11 = self._compute_square_chol(forward_var, layer.hidden_layer_size)
        e22 = self._compute_square_chol(backward_var, layer.hidden_layer_size)
        e12 = Tensor.dot(forward_centered, backward_centered.T)

        corr = Tensor.dot(Tensor.dot(e11, e12), e22)

        loss = Tensor.sqrt(Tensor.sum(corr))

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
                         givens={var_x: training_x[index * hyper_parameters.batch_size:
                                                            (index + 1) * hyper_parameters.batch_size, :],
                                 var_y: training_y[index * hyper_parameters.batch_size:
                                                            (index + 1) * hyper_parameters.batch_size, :]})

                #Calculating number of batches
        n_training_batches = training_x.get_value(borrow=True).shape[0] / hyper_parameters.batch_size

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

    def _add_cross_encoder_layer(self, layer_size, symmetric_double_encoder, activation_hidden, activation_output):

        layer_count = len(symmetric_double_encoder)

        symmetric_layer = SymmetricHiddenLayer(numpy_range=self._random_range,
                                               hidden_layer_size=layer_size,
                                               name="layer" + str(layer_count),
                                               activation_hidden=activation_hidden,
                                               activation_output=activation_output)

        symmetric_double_encoder.add_hidden_layer(symmetric_layer)

    def set_parameters(self, parameters):
        return

    def _compute_square_chol(self, a, n):

        w, v = Tensor.nlinalg.eigh(a,'L')

        result, updates = scan(lambda eigs, eigv, prior_results, size: Tensor.sqrt(eigs) * Tensor.dot(eigv.reshape([size, 1]), eigv.reshape([1, size])),
                               outputs_info=Tensor.zeros_like(a),
                               sequences=[w, v.T],
                               non_sequences=n)

        return result[-1]


