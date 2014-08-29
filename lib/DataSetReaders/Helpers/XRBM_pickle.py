import cPickle
import gzip

import numpy

from lib.DataSetReaders.XRBM_dataset import XRBMRawDataSet


if __name__ == '__main__':

    dataset_path = '/home/aviv/Project/DoubleEncoder/DataSet/XRBM/JW11_fold0'
    dataset = XRBMRawDataSet(dataset_path)

    test_set = dataset.testset[0][:, 0:99], dataset.testset[1][:, 0:99]

    training_set_x1 = numpy.concatenate([dataset.trainset[0], dataset.tuning[0]], axis=1)
    training_set_x2 = numpy.concatenate([dataset.trainset[1], dataset.tuning[1]], axis=1)

    training_set = training_set_x1[:, 0:999], training_set_x2[:, 0:999]

    f = gzip.open('./XRBM.small.gzip.pkl', mode='w+')

    cPickle.dump((training_set, test_set), f)

    f.close()