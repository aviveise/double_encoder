import os
import pickle
import numpy

from theano.tensor import config, theano

from lib.MISC.container import ContainerRegisterMetaClass
from dataset_base import DatasetBase

TRAINING_PERCENT = 0.8


class ARIKDataSet(DatasetBase):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(ARIKDataSet, self).__init__(data_set_parameters)

    def build_dataset(self):
        print "*****************************************************************"

        type_A_train_path = os.path.join(self.dataset_path,
                                         'normalized_and_avg_subtract_binary_validation_labels.pickled')
        type_B_train_path = os.path.join(self.dataset_path,
                                         'normalized_and_avg_subtract_raw_predicted_validation_labels.pickled')
        type_A_test_path = os.path.join(self.dataset_path, 'all_binary_normalized_and_avg_subtract_values.pickled')
        type_B_test_path = os.path.join(self.dataset_path,
                                        'raw_predicted_test_labels_normalized_and_avg_subtract_values.pickled')

        with file(type_A_train_path, 'r') as type_A_file:
            type_A_train = pickle.load(type_A_file)

        with file(type_B_train_path, 'r') as type_B_file:
            type_B_train = pickle.load(type_B_file)

        with file(type_A_test_path, 'r') as type_A_file:
            type_A_test = pickle.load(type_A_file)

        with file(type_B_test_path, 'r') as type_B_file:
            type_B_test = pickle.load(type_B_file)

        A_train_set, A_tuning_set, test_samples = self.produce_optimization_sets(type_A_train)
        B_train_set, B_tuning_set, test_samples = self.produce_optimization_sets(type_B_train, test_samples)

        self.trainset = [numpy.cast[theano.config.floatX](A_train_set), numpy.cast[theano.config.floatX](B_train_set)]

        self.tuning = [numpy.cast[theano.config.floatX](A_tuning_set), numpy.cast[theano.config.floatX](B_tuning_set)]

        self.testset = [numpy.cast[theano.config.floatX](type_A_test), numpy.cast[theano.config.floatX](type_B_test)]

        if not self.testset[0].shape[0] == self.testset[1].shape[0]:
            size = min(self.testset[1].shape[0], self.testset[0].shape[0])
            self.testset[0] = self.testset[0][0:size, :]
            self.testset[1] = self.testset[1][0:size, :]
