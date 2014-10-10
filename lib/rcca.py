__author__ = 'aviv'

import sys
import ConfigParser
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import DataSetReaders
import numpy
import matplotlib.pyplot as plt

from Testers.trace_correlation_tester import TraceCorrelationTester
from Transformers.identity_transform import IdentityTransformer
from MISC.container import Container
from MISC.utils import ConfigSectionMap

__author__ = 'aviv'

if __name__ == '__main__':

    data_set_config = sys.argv[1]
    rca_location = sys.argv[2]
    kx = int(sys.argv[3])
    ky = int(sys.argv[4])
    top = int(sys.argv[5])

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    #construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)

    robjects.r('source("%s")' % rca_location)

    mnist = robjects.r('load')('/home/aviv/Project/DoubleEncoder/DataSet/MNIST_SPLIT/mnist.data')

    x1 = numpy.array(robjects.r['x_tr'])
    x2 = numpy.array(robjects.r['y_tr'])

    x1_test = numpy.array(robjects.r['x_te'])
    x2_test = numpy.array(robjects.r['y_te'])

    plt.imshow(x1_test.T[:, 1].reshape([14, 28]))
    plt.gray()
    plt.show()

    plt.imshow(x2_test.T[:, 1].reshape([14, 28]))
    plt.gray()
    plt.show()

    print 'training rcca'

    rcca_fit = robjects.r('rcca_fit')(x=numpy2ri.numpy2ri(data_set.trainset[1].T),
                                      y=numpy2ri.numpy2ri(data_set.trainset[0].T),
                                      kx=kx,
                                      ky=ky,
                                      top=top,
                                      type='nystrom')

    print 'evaluating rcca'

    rcca_eval = robjects.r('rcca_eval')(rcca=rcca_fit,
                                        x=numpy2ri.numpy2ri(data_set.testset[1].T),
                                        y=numpy2ri.numpy2ri(data_set.testset[0].T))

    x = numpy.array(rcca_eval[0])
    y = numpy.array(rcca_eval[1])

    trace_correlation = TraceCorrelationTester(x, y).test(IdentityTransformer())

    print '\nResults:\n'

    print 'trace: correlation\n'

    print '%f%%\n' % float(trace_correlation)



    print 'done'

