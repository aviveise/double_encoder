import os
import sys
import struct

import numpy
import scipy.io

from theano import config

from DataSetReaders.dataset_factory import dataset_meta
from DataSetReaders.dataset_base import DatasetBase

TRAINING_PERCENT = 0.8

class DRUGSDataSet(DatasetBase):

    __metaclass__ = dataset_meta

    def __init__(self, dataset_path, center, normalize, whiten):
        super(DRUGSDataSet, self).__init__(dataset_path, 'DRUGS', center, normalize, whiten)

    def build_dataset(self):

        mat = scipy.io.loadmat(self.dataset_path)

        fingerprints = mat.get('fingerprints')
        side_effects = mat.get('side_effects')

        drug_number = fingerprints.shape[0]

        training_size = int(drug_number * TRAINING_PERCENT * 0.9)
        tuning_size = int(drug_number * TRAINING_PERCENT * 0.1)
        test_size = int(drug_number * (1 - TRAINING_PERCENT))

        self.trainset = [fingerprints[0: training_size, :].T.astype(config.floatX, copy=False),
                         side_effects[0: training_size, :].T.astype(config.floatX, copy=False)]

        self.tuning = [fingerprints[training_size: training_size + tuning_size, :].T.astype(config.floatX, copy=False),
                       side_effects[training_size: training_size + tuning_size, :].T.astype(config.floatX, copy=False)]

        self.testset = [fingerprints[training_size + tuning_size: training_size + tuning_size + test_size, :].T.astype(config.floatX, copy=False),
                        side_effects[training_size + tuning_size: training_size + tuning_size + test_size, :].T.astype(config.floatX, copy=False)]

