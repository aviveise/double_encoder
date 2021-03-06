import scipy.io

from theano import config

from lib.MISC.container import ContainerRegisterMetaClass
from dataset_base import DatasetBase

TRAINING_PERCENT = 0.9

class DRUGSDataSet(DatasetBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(DRUGSDataSet, self).__init__(data_set_parameters)

    def build_dataset(self):

        mat = scipy.io.loadmat(self.dataset_path)

        fingerprints = mat.get('fingerprints')
        side_effects = mat.get('side_effects')

        drug_number = fingerprints.shape[0]

        training_size = int(drug_number * TRAINING_PERCENT * 0.95)
        tuning_size = int(drug_number * TRAINING_PERCENT * 0.05)
        test_size = int(drug_number * (1 - TRAINING_PERCENT))

        self.trainset = [fingerprints[0: training_size, :].astype(config.floatX, copy=False),
                         side_effects[0: training_size, :].astype(config.floatX, copy=False)]

        self.tuning = [fingerprints[training_size: training_size + tuning_size, :].astype(config.floatX, copy=False),
                       side_effects[training_size: training_size + tuning_size, :].astype(config.floatX, copy=False)]

        self.testset = [fingerprints[training_size + tuning_size: training_size + tuning_size + test_size, :].astype(config.floatX, copy=False),
                        side_effects[training_size + tuning_size: training_size + tuning_size + test_size, :].astype(config.floatX, copy=False)]

        self.x_y_mapping = {'train': side_effects[0: training_size, :],
                            'dev': side_effects[training_size: training_size + tuning_size, :],
                            'test': side_effects[training_size + tuning_size: training_size + tuning_size + test_size, :]}