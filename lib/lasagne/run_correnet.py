from math import floor
import matplotlib
import traceback
from scipy.spatial.distance import cdist
from sklearn import preprocessing

from lib.Model.corrnet import trainCorrNet2
from lib.MISC.utils import calculate_mardia

matplotlib.use('Agg')

import ConfigParser
import os
import sys
import cPickle
import lib.lasagne
import numpy
import json
from collections import OrderedDict
from tabulate import tabulate
from theano import tensor, theano
from lib.MISC.container import Container
from lib.MISC.logger import OutputLog
from lib.MISC.utils import ConfigSectionMap
import lib.DataSetReaders

OUTPUT_DIR = r'C:\Workspace\output'
VALIDATE_ALL = False
TEST=False

if __name__=='__main__':
    data_set_config = sys.argv[1]
    if len(sys.argv) > 2:
        top = int(sys.argv[2])
    else:
        top = 0

    if TEST:
        view_x = numpy.load(open(os.path.join(OUTPUT_DIR,'view_x.npy'),'r'))
        view_y = numpy.load(open(os.path.join(OUTPUT_DIR,'view_y.npy'),'r'))

        print 'correlation: {0}'.format(calculate_mardia(view_x, view_y, top=0))


    model_results = {'train': [], 'validate': []}

    OutputLog().set_path(OUTPUT_DIR)
    OutputLog().set_verbosity('info')

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)
    data_set.load()

    batch_size = 100
    training_epochs = 50
    l_rate = 0.01
    optimization = "rmsprop"
    tied = True
    n_visible_left = 112
    n_visible_right = 273
    n_hidden = 112
    lamda = 2
    hidden_activation = "sigmoid"
    output_activation = "sigmoid"
    loss_fn = "squarrederror"


    trainCorrNet2(data_set.trainset[0],data_set.trainset[1], data_set.testset[0], data_set.testset[1],batch_size=batch_size,
                  training_epochs=training_epochs,l_rate=l_rate,optimization=optimization,tied=tied,
                  n_visible_left=n_visible_left,n_visible_right=n_visible_right,n_hidden=n_hidden,hidden_activation=hidden_activation,
                  output_activation=output_activation)