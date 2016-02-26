import os
import pickle
import sys
import ConfigParser
import gc
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import STAP
import numpy
from tabulate import tabulate

from Testers.trace_correlation_tester import TraceCorrelationTester
from Transformers.identity_transform import IdentityTransformer
from MISC.utils import ConfigSectionMap, complete_rank, calculate_mardia
from MISC.container import Container

import DataSetReaders
from configuration import Configuration
from MISC.logger import OutputLog

__author__ = 'aviv'

OUTPUT_DIR = r'C:\Workspace\output'

if __name__ == '__main__':
    data_set_config = sys.argv[1]
    rca_location = sys.argv[3]
    kx = int(sys.argv[4])
    ky = int(sys.argv[5])
    top = int(sys.argv[6])

    # parse runtime configuration
    OutputLog().set_path(OUTPUT_DIR)

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)

    # robjects.r('source("%s")' % rca_location)

    with open(rca_location) as rca_file:
        rca_string = rca_file.read()

    rcca_fit = STAP(rca_string, "rcca_fit")
    rcca_eval = STAP(rca_string, "rcca_eval")

    # mnist = robjects.r('load')('/home/aviv/Project/DoubleEncoder/DataSet/MNIST_SPLIT/mnist.data')

    # x1_rcca = numpy.array(robjects.r['x_tr'])
    # x2_rcca = numpy.array(robjects.r['y_tr'])
    #
    # x1_dataset = data_set.trainset[1]
    # x2_dataset = data_set.trainset[0]
    #
    # x1_test = numpy.array(robjects.r['x_te'])
    # x2_test = numpy.array(robjects.r['y_te'])


    OutputLog().write('training rcca')

    results = rcca_fit.rcca_fit(x=numpy2ri.numpy2ri(data_set.trainset[1]),
                                y=numpy2ri.numpy2ri(data_set.trainset[0]),
                                kx=kx,
                                ky=ky,
                                top=top,
                                type='nystrom')

    gc.collect()

    # rcca_fit = robjects.r('rcca_fit')(x=numpy2ri.numpy2ri(data_set.trainset[1].T),
    #                                   y=numpy2ri.numpy2ri(data_set.trainset[0].T),
    #                                   kx=kx,
    #                                   ky=ky,
    #                                   top=top,
    #                                   type='nystrom')
    #
    print 'evaluating rcca'
    #
    eval_results = rcca_eval.rcca_eval(rcca=results,
                                       x=numpy2ri.numpy2ri(data_set.testset[1]),
                                       y=numpy2ri.numpy2ri(data_set.testset[0]))

    gc.collect()

    # rcca_eval = robjects.r('rcca_eval')(rcca=rcca_fit,
    #                                     x=numpy2ri.numpy2ri(data_set.testset[1].T),
    #                                     y=numpy2ri.numpy2ri(data_set.testset[0].T))
    #
    x = numpy.array(eval_results[0])
    y = numpy.array(eval_results[1])
    #

    OutputLog().write('Output shape: {0}'.format(x.shape))

    header = ['layer', 'corr', 'search1', 'search5', 'search10', 'search_sum', 'desc1', 'desc5', 'desc10',
              'desc_sum']

    rows=[]

    correlation = calculate_mardia(x, y, top)
    search_recall, describe_recall = complete_rank(x, y, data_set.reduce_val)

    print_row = ["rcca", correlation]
    print_row.extend(search_recall)
    print_row.append(sum(search_recall))
    print_row.extend(describe_recall)
    print_row.append(sum(describe_recall))

    rows.append(print_row)

    OutputLog().write(tabulate(rows, headers=header))

    OutputLog().write('done')

    pickle.dump(x,open(os.path.join(OutputLog().output_path, 'test_x.p', 'rb')))
    pickle.dump(y,open(os.path.join(OutputLog().output_path, 'test_y.p', 'rb')))
