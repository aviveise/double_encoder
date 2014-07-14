__author__ = 'aviv'

import os
import sys
import datetime

from singleton import Singleton
from enum import enum

class OutputLog(object):

    __metaclass__= Singleton

    def __init__(self):
        output_file_name = 'double_encoder_' + str(datetime.datetime.now()) + '.txt'
        self.output_file = open(output_file_name, 'w+')

    def write(self, message):
        self.output_file.write(message + '\n')
        self.output_file.flush()

    def __del__(self):
        self.output_file.close()