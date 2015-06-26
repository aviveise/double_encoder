import os
import sys

__author__ = 'aviv'

class Interval(object):

    def __init__(self, start_value, end_value, step_value):

        self.start = start_value
        self.end = end_value
        self.step = step_value