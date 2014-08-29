import gzip
import cPickle

from lib.DataSetReaders.JAPAN_dataset import JAPANDataSetVideo
__author__ = 'aviv'


if __name__ == '__main__':

    a = JAPANDataSetVideo('/home/aviv/Project/DoubleEncoder/DataSet/JAPAN/LBM2_SCN1.avi')

    f = gzip.open('/home/aviv/Project/DoubleEncoder/bin/japan.gzip.pkl', mode='w+')

    cPickle.dump((a.trainset, a.tuning, a.testset), f)

    f.close()