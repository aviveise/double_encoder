import os
import sys

from MISC.singleton import Singleton
from MISC.factory_base import FactoryBase

__author__ = 'aviv'

class dataset_meta(type):

    def __init__(cls, name, bases, attr):
        DatasetFactory().register(name, cls)

class DatasetFactory(FactoryBase):

    __metaclass__ = Singleton

    def __init__(self):

        super(DatasetFactory,self).__init__()