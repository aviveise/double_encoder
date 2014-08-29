__author__ = 'aviv'

from lib.MISC.utils import print_list

class HyperParameters(object):

    def __init__(self, layer_sizes = [0],
                       learning_rate = 0,
                       batch_size = 0,
                       epochs = 0,
                       momentum = 0):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.layer_sizes = layer_sizes

    def copy(self):

        return HyperParameters(self.layer_sizes,
                               self.learning_rate,
                               self.batch_size,
                               self.epochs,
                               self.momentum)

    def print_parameters(self, output_stream):

        output_stream.write('Hyperparameters:')

        output_stream.write('layer_sizes : %s\n'
                            'learning_rate: %f\n'
                            'batch_size: %d\n'
                            'epochs: %d\n'
                            'Momentum: %f\n' % (print_list(self.layer_sizes),
                                                self.learning_rate,
                                                self.batch_size,
                                                self.epochs,
                                                self.momentum))
