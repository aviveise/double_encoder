__author__ = 'aviv'

import sys
import os

import theano.tensor as Tensor
import abc

from numpy.random import RandomState
from theano import shared
from trainer import Trainer

class TrainingStrategy(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, training_set_x, training_set_y, hyper_parameters):

        #Training inputs x1 and x2 as a matrices with columns as samples
        self._x = Tensor.matrix('x')
        self._y = Tensor.matrix('y')

        #need to convert the input into tensor variable
        self._training_set_x = shared(training_set_x, 'training_set_x', borrow=True)
        self._training_set_y = shared(training_set_y, 'training_set_y', borrow=True)

        self._hyper_parameters = hyper_parameters
        self._random_range = RandomState()
        self._symmetric_double_encoder = None

        regularization_methods = self.get_regularization_methods()

        self._trainer = Trainer(self._x, self._y, self._training_set_x, self._training_set_y,
                                self._hyper_parameters, regularization_methods)

    @abc.abstractmethod
    def get_regularization_methods(self):
        return

    @abc.abstractmethod
    def start_training(self):
        return

    def get_double_encoder(self):
        return self._symmetric_double_encoder