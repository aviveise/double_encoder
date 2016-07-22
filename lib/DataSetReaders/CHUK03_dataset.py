import os
import scipy.io
import h5py
import numpy

from theano.tensor import config, theano

from lib.MISC.container import ContainerRegisterMetaClass
from dataset_base import DatasetBase

TRAINING_PERCENT = 0.9


class CHUK03DataSet(DatasetBase):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(CHUK03DataSet, self).__init__(data_set_parameters)

    def build_dataset(self):

        dataset_mat = h5py.File(os.path.join(self.dataset_path, 'cuhk-03.mat'), 'r')

        training_images_x = []
        training_images_y = []

        detected_identities = self._get_identities(dataset_mat, dataset_mat['detected'][0])
        labeled_identities = self._get_identities(dataset_mat, dataset_mat['labeled'][0])

        test_identities = {
            'identities': [dataset_mat[group][1] for group in dataset_mat['testsets'][0]],
            'camera_pairs': [dataset_mat[group][0] for group in dataset_mat['testsets'][0]]
        }

        test_identities['identities'] = self._flatten(test_identities['identities'])
        test_identities['camera_pairs'] = self._flatten(test_identities['camera_pairs'])



    def _get_identities(self, dataset_mat, data):
        identities = []
        for cameras_data in data:
            parsed_data = dataset_mat[cameras_data]
            for identity_index in range(dataset_mat[cameras_data].shape[1]):
                identity = {
                    'camera_A': [numpy.array(dataset_mat[parsed_data[camera_index, identity_index]]) for
                                 camera_index in range(5)],
                    'camera_B': [numpy.array(dataset_mat[parsed_data[camera_index, identity_index]]) for
                                 camera_index in range(5,10)]
                }
                identities.append(identity)

        return identities

    def _flatten(self, list):
        return [item for sublist in list for item in sublist]


