import os
import scipy.io

from theano import config

from MISC.container import ContainerRegisterMetaClass
from dataset_base import DatasetBase

TRAINING_PERCENT = 0.8

class CUAVEDataSet(DatasetBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(CUAVEDataSet, self).__init__(data_set_parameters)

    def build_dataset(self):

        onlyfiles = [f for f in os.listdir(self.dataset_path) if os.path.isfile(os.path.join(self.dataset_path,f))]

        for f in onlyfiles:
            mat = scipy.io.loadmat(self.dataset_path + '/' + f)
            first_person = mat['video'][1, 0]
            second_person = mat['video'][1, 1]



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

