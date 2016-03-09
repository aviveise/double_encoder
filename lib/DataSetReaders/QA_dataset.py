import os
import scipy.io
import h5py
import numpy
from theano.tensor import config, theano
from lib.MISC.container import ContainerRegisterMetaClass
from dataset_base import DatasetBase

TRAINING_PERCENT = 0.8


class QADataSet(DatasetBase):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(QADataSet, self).__init__(data_set_parameters)

    def build_dataset(self):
        info_mat = h5py.File(os.path.join(self.dataset_path, 'dataset_info.mat'), 'r')
        sent_mat = h5py.File(os.path.join(self.dataset_path, 'sents.mat'), 'r')

        training_sents = sent_mat[sent_mat['sent_vecs'][1][0]]
        validation_sents = sent_mat[sent_mat['sent_vecs'][2][0]]
        testing_sents = sent_mat[sent_mat['sent_vecs'][3][0]]

        trn_number_of_questions = len(training_sents)
        dev_number_of_questions = len(validation_sents)
        tst_number_of_questions = len(testing_sents)

        trn_number_of_pairs = info_mat['num_valid_pairs'][1][0]
        dev_number_of_pairs = info_mat['num_valid_pairs'][2][0]
        tst_number_of_pairs = info_mat['num_valid_pairs'][3][0]

        q = numpy.where(info_mat[info_mat['valid_cell'][0][0]][0] == 1)[0][0]
        dim = sent_mat[training_sents[q][0]]['q_vec'].shape[1]

        self.trainset = [numpy.ndarray((trn_number_of_pairs, dim), dtype=config.floatX),
                         numpy.ndarray((trn_number_of_pairs, dim), dtype=config.floatX)]

        self.tuning = [numpy.ndarray((dev_number_of_pairs, dim), dtype=config.floatX),
                       numpy.ndarray((dev_number_of_pairs, dim), dtype=config.floatX)]

        self.testset = [numpy.ndarray((tst_number_of_pairs, dim), dtype=config.floatX),
                        numpy.ndarray((tst_number_of_pairs, dim), dtype=config.floatX)]

        self.x_y_mapping = {'train': numpy.zeros((trn_number_of_questions, trn_number_of_pairs )),
                            'dev': numpy.zeros((dev_number_of_questions, dev_number_of_pairs)),
                            'test': numpy.zeros((tst_number_of_questions, tst_number_of_pairs))}

        current_pair = 0
        for i in range(trn_number_of_questions):

            if 'q_vec' not in sent_mat[training_sents[i][0]]:
                continue

            question = sent_mat[training_sents[i][0]]['q_vec'][0]
            answares = sent_mat[training_sents[i][0]]['a_vecs']

            for index, answare in enumerate(answares):
                self.trainset[0][index, :] = question.T
                self.trainset[1][index, :] = answare.T
                self.x_y_mapping['train'][i, current_pair + index] = info_mat[info_mat[info_mat['sets_data'][1][0]][0][i]]['labels'][index][0]

            current_pair += index

        current_pair = 0
        for i in range(dev_number_of_questions):

            if 'q_vec' not in sent_mat[validation_sents[i][0]]:
                continue

            question = sent_mat[validation_sents[i][0]]['q_vec'][0]
            answares = sent_mat[validation_sents[i][0]]['a_vecs']

            for index, answare in enumerate(answares):
                self.tuning[0][index, :] = question.T
                self.tuning[1][index, :] = answare.T
                self.x_y_mapping['dev'][i, current_pair + index] = info_mat[info_mat[info_mat['sets_data'][1][0]][0][i]]['labels'][index][0]

            current_pair += index

        current_pair = 0
        for i in range(tst_number_of_pairs):

            if 'q_vec' not in sent_mat[testing_sents[i][0]]:
                continue

            question = sent_mat[training_sents[i][0]]['q_vec'][0]
            answares = sent_mat[training_sents[i][0]]['a_vecs']

            for index, answare in enumerate(answares):
                self.testset[0][index, :] = question.T
                self.testset[1][index, :] = answare.T
                self.x_y_mapping['test'][i, current_pair + index] = info_mat[info_mat[info_mat['sets_data'][1][0]][0][i]]['labels'][index][0]

            current_pair += index

    def jitter(self, set):
        new_samples = set + numpy.cast[theano.config.floatX](numpy.random.normal(0, 0.25, set.shape))
        return new_samples
