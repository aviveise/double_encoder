import os
import sys
import abc

from MISC.singleton import Singleton
from MISC.factory_base import FactoryBase

__author__ = 'aviv'

class OptimizationMetaClass(abc.ABCMeta):

    def __init__(cls, name, bases, attr):
        OptimizationFactory().register(name, cls)

class OptimizationFactory(FactoryBase):

    __metaclass__ = Singleton

    def __init__(self):

        super(OptimizationFactory,self).__init__()