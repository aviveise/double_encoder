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
        self.reduce_test = 1

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

        trn_valid = info_mat[info_mat['valid_cell'][1][0]][0]
        dev_valid = info_mat[info_mat['valid_cell'][2][0]][0]
        tst_valid = info_mat[info_mat['valid_cell'][3][0]][0]

        q = numpy.where(info_mat[info_mat['valid_cell'][1][0]][0] == 1)[0][0]
        dim = sent_mat[training_sents[q][0]]['q_vec'].shape[1]

        self.trainset = [numpy.ndarray((trn_number_of_pairs, dim), dtype=config.floatX),
                         numpy.ndarray((trn_number_of_pairs, dim), dtype=config.floatX)]

        self.tuning = [numpy.ndarray((dev_number_of_pairs, dim), dtype=config.floatX),
                       numpy.ndarray((dev_number_of_pairs, dim), dtype=config.floatX)]

        self.testset = [numpy.ndarray((tst_number_of_pairs, dim), dtype=config.floatX),
                        numpy.ndarray((tst_number_of_pairs, dim), dtype=config.floatX)]

        self.x_y_mapping = {
            'train': numpy.ndarray((0, trn_number_of_pairs)),
            'dev': numpy.ndarray((0, dev_number_of_pairs)),
            'test': numpy.ndarray((0, tst_number_of_pairs)),
        }

        self.x_reduce = {
            'train': [],
            'dev': [],
            'test': []
        }

        current_pair = 0
        q_index = 0
        not_valid = numpy.nonzero(trn_valid == 0)[0] + 1
        for i in range(trn_number_of_questions):

            if 'q_vec' not in sent_mat[training_sents[i][0]] or i in not_valid:
                continue

            self.x_y_mapping['train'] = numpy.vstack((self.x_y_mapping['train'], numpy.zeros((1, trn_number_of_pairs))))

            question = sent_mat[training_sents[i][0]]['q_vec'][0]
            answares = numpy.array(sent_mat[training_sents[i][0]]['a_vecs'])

            self.x_reduce['train'].append(current_pair)
            # for index, answare in enumerate(answares):
            #     # if index == 5:
            #     #     break
            #     self.trainset[0][index + current_pair, :] = question.T
            #     self.trainset[1][index + current_pair, :] = answare.T
            #     self.x_y_mapping['train'][q_index, index + current_pair]


            self.trainset[0][q_index, :] = question.T
            self.trainset[1][q_index, :] = numpy.mean(answares.T, axis=1)
            self.x_y_mapping['train'][q_index, q_index] = 1

            q_index += 1
            # current_pair += index + 1

        # self.trainset = [self.trainset[0][0: current_pair + 1], self.trainset[1][0: current_pair + 1]]
        # self.x_y_mapping['train'] = self.x_y_mapping['train'][:, 0: current_pair + 1]

        self.trainset = [self.trainset[0][0: q_index], self.trainset[1][0: q_index]]
        self.x_y_mapping['train'] = self.x_y_mapping['train'][:, 0: q_index]

        current_pair = 0
        q_index = 0
        not_valid = numpy.nonzero(dev_valid == 0)[0] + 1
        for i in range(dev_number_of_questions):

            if 'q_vec' not in sent_mat[validation_sents[i][0]] or i in not_valid:
                continue

            self.x_y_mapping['dev'] = numpy.vstack((self.x_y_mapping['dev'], numpy.zeros((1, dev_number_of_pairs))))

            question = sent_mat[validation_sents[i][0]]['q_vec'][0]
            answares = sent_mat[validation_sents[i][0]]['a_vecs']

            self.x_reduce['dev'].append(current_pair)
            for index, answare in enumerate(answares):
                self.tuning[0][index + current_pair, :] = question.T
                self.tuning[1][index + current_pair, :] = answare.T
                self.x_y_mapping['dev'][q_index, index + current_pair] = 1

            q_index += 1
            current_pair += index + 1

        self.tuning = [self.tuning[0][0:current_pair + 1], self.tuning[1][0:current_pair + 1]]
        self.x_y_mapping['dev'] = self.x_y_mapping['dev'][:, 0: current_pair + 1]

        current_pair = 0
        q_index = 0
        not_valid = numpy.nonzero(tst_valid == 0)[0] + 1
        for i in range(tst_number_of_questions):

            if 'q_vec' not in sent_mat[testing_sents[i][0]] or i in not_valid:
                continue

            self.x_y_mapping['test'] = numpy.vstack((self.x_y_mapping['test'], numpy.zeros((1, tst_number_of_pairs))))

            question = sent_mat[testing_sents[i][0]]['q_vec'][0]
            answares = sent_mat[testing_sents[i][0]]['a_vecs']

            self.x_reduce['test'].append(current_pair)
            for index, answare in enumerate(answares):
                self.testset[0][index + current_pair, :] = question.T
                self.testset[1][index + current_pair, :] = answare.T
                self.x_y_mapping['test'][q_index, index + current_pair] = 1

            q_index += 1
            current_pair += index + 1

        self.testset = [self.testset[0][0:current_pair + 1], self.testset[1][0:current_pair + 1]]
        self.x_y_mapping['test'] = self.x_y_mapping['test'][:, 0: current_pair + 1]

    def jitter(self, set):
        new_samples = set + numpy.cast[theano.config.floatX](numpy.random.normal(0, 0.25, set.shape))
        return new_samples
