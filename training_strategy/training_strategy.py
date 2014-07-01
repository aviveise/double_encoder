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

    def __init__(self):
        self._random_range = RandomState()

    @abc.abstractmethod
    def train(self, training_set_x, training_set_y, hyper_parameters, regularization_methods, activation_method):
        return
