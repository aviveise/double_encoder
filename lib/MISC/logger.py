__author__ = 'aviv'

import datetime
import os

from singleton import Singleton

verbosity_map = {
    'info': 1,
    'debug': 2,
}

class OutputLog(object):

    __metaclass__ = Singleton

    def __init__(self):
        self.output_file_name = 'double_encoder_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.txt'
        self._verbosity = verbosity_map['info']

    def write(self, message, verbosity='info'):
        if verbosity_map[verbosity] <= self._verbosity:
            print message
            self.output_file.write(message + '\n')
            self.output_file.flush()

    def set_path(self, path):
        self.path = path
        self.output_file = open(os.path.join(self.path, self.output_file_name), 'w+')

    def __del__(self):
        self.output_file.close()

    def set_verbosity(self, verbosity):
        self._verbosity = verbosity_map[verbosity]