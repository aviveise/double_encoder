__author__ = 'aviv'

import abc

from numpy.random import RandomState


class TrainingStrategy(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._random_range = RandomState()

    def train(self,
              training_set_x,
              training_set_y,
              hyper_parameters,
              regularization_methods,
              activation_method,
              top=50,
              print_verbose=False,
              validation_set_x=None,
              validation_set_y=None,
              import_net=False,
              import_path='',
              reduce_val=0):
        return

    def set_parameters(self, parameters):
        return
