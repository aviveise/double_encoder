from MISC import utils

from DataSetReaders.XRBM_dataset import XRBMDataSet

__author__ = 'aviv'


if __name__ == '__main__':

    a = XRBMDataSet('/home/aviv/Project/DoubleEncoder/DataSet/XRBM/JW11_fold0', False, False, True)
    utils.testWhitenTransform(a.trainset[0])
    utils.testWhitenTransform(a.trainset[1])
