import os
import sys
import gzip
import cPickle

from DataSetReaders.FISH_dataset import FISHDataSet
__author__ = 'aviv'


if __name__ == '__main__':

    a = FISHDataSet('/home/aviv/Project/DoubleEncoder/DataSet/FISH/', False, False, False)