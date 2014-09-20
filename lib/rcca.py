__author__ = 'aviv'

import sys
import ConfigParser
import rpy2.robjects.numpy2ri as numpy2ri
import rpy2.robjects as robjects
import DataSetReaders

from MISC.container import Container
from MISC.utils import ConfigSectionMap

__author__ = 'aviv'

def GetMatrix(data):

    matrix = robjects.r['matrix'](0, data.shape[0], data.shape[1])

    for i in xrange(data.shape[0]):
        floatVector = robjects.FloatVector(data_set.trainset[0][1, :])
        matrix.rx[i, :] = floatVector

if __name__ == '__main__':

    data_set_config = sys.argv[1]
    rca_location = sys.argv[2]

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    #construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)

    robjects.r('source("%s")' % rca_location)

    print 'training rcca'

    rcca_fit = robjects.r('rcca_fit')(x=numpy2ri.numpy2ri(data_set.trainset[0].T),
                                      y=numpy2ri.numpy2ri(data_set.trainset[1].T),
                                      kx=data_set.trainset[0].shape[0],
                                      ky=data_set.trainset[1].shape[0],
                                      top=112,
                                      type='nystrom')

    print 'evaluating rcca'

    rcca_eval = robjects.r('rcca_eval')(rcca=rcca_fit,
                                        x=numpy2ri.numpy2ri(data_set.testset[0].T),
                                        y=numpy2ri.numpy2ri(data_set.testset[1].T))

    print 'done'

    rcca_cor = robjects.r('rcca_cor')(rcca_eval=rcca_eval,top=112)