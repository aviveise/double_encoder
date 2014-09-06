__author__ = 'aviv'

import abc

from numpy.random import RandomState


class TrainingStrategy(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._random_range = RandomState()

    @abc.abstractmethod
    def train(self,
              training_set_x,
              training_set_y,
              hyper_parameters,
              regularization_methods,
              activation_method):
        return

    @abc.abstractmethod
    def _set_parameters(self, parameters):
        return
