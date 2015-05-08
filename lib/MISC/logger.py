__author__ = 'aviv'

import datetime
import os

from singleton import Singleton


class OutputLog(object):

    __metaclass__ = Singleton

    def __init__(self):
        self.output_file_name = 'double_encoder_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.txt'

    def write(self, message):
        print message
        self.output_file.write(message + '\n')
        self.output_file.flush()

    def set_path(self, path):
        self.path = path
        self.output_file = open(os.path.join(self.path, self.output_file_name), 'w+')

    def __del__(self):
        self.output_file.close()