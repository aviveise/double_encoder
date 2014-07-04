import os
import sys
import abc

from MISC.singleton import Singleton

__author__ = 'aviv'


class ContainerRegisterMetaClass(abc.ABCMeta):

    def __init__(cls, name, bases, attr):
        Container().register(name, cls)

class Container(object):

    __metaclass__ = Singleton

    def __init__(self):

        self.items = {}

    def register(self, name, type):

        self.items[name] = type

    def create(self, name, *args):

        item_type = self.items[name]
        return item_type(*args)