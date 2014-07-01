__author__ = 'aviv'

import os
import sys
import datetime

from singleton import Singleton
from enum import enum

Verbosity = enum(QUITE=1, VERBOSE=2, DEBUG=3)

class Logger(object):

    __metaclass__= Singleton

    def __init__(self):
        output_file_name = 'double_encoder_' + str(datetime.datetime.now()) + '.txt'
        self.output_file = open(output_file_name, 'w+')
        self.verbosity = Verbosity.QUITE

    def set_verbosity(self, verbosity):
        self.verbosity = verbosity

    def write(self, message, verbosity):
        if verbosity >= self.verbosity:
            self.output_file.write(message)

    def __del__(self):
        self.output_file.close()