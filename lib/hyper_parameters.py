__author__ = 'aviv'

import theano.tensor as Tensor

from MISC.utils import print_list


class HyperParameters(object):
    def __init__(self, layer_sizes=[0],
                 learning_rate=0,
                 batch_size=0,
                 epochs=0,
                 momentum=0,
                 method_in=Tensor.nnet.sigmoid,
                 method_out=Tensor.nnet.sigmoid,
                 training_strategy='SGD',
                 rho=0.5,
                 cascade_train=True):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.layer_sizes = layer_sizes
        self.method_in = method_in
        self.method_out = method_out
        self.training_strategy = training_strategy
        self.rho = rho
        self.cascade_train = cascade_train

    def copy(self):
        return HyperParameters(self.layer_sizes,
                               self.learning_rate,
                               self.batch_size,
                               self.epochs,
                               self.momentum,
                               self.method_in,
                               self.method_out)

    def print_parameters(self, output_stream):
        output_stream.write('Hyperparameters:')

        output_stream.write('layer_sizes : %s\n'
                            'learning_rate: %f\n'
                            'batch_size: %d\n'
                            'epochs: %d\n'
                            'Momentum: %f\n'
                            'method_in: %s\n'
                            'method_out: %s \n'
                            'training strgy: %s \n'
                            'rho: %f \n'
                            'cascade_train: %r \n' % (print_list(self.layer_sizes),
                                                      self.learning_rate,
                                                      self.batch_size,
                                                      self.epochs,
                                                      self.momentum,
                                                      self.method_in.__str__(),
                                                      self.method_out.__str__(),
                                                      self.training_strategy,
                                                      self.rho,
                                                      self.cascade_train))
