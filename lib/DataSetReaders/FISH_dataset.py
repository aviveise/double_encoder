import os
import numpy
import scipy.io
import struct
from theano import config
from lib.MISC import utils
from lib.MISC.container import ContainerRegisterMetaClass
from dataset_base import DatasetBase

TRAINING_PERCENT = 0.9


class FISHDataSet(DatasetBase):
    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, data_set_parameters):
        super(FISHDataSet, self).__init__(data_set_parameters)

    def build_dataset(self):

        X, Y = self.load_fish_database(os.path.join(self.dataset_path, 'fish_database.mat'),
                                       os.path.join(self.dataset_path, 'fasta.fas'))

        training_size = int(X.shape[0] * TRAINING_PERCENT * 0.9)
        tuning_size = int(X.shape[0] * TRAINING_PERCENT * 0.1)
        test_size = int(X.shape[0] * (1 - TRAINING_PERCENT))

        self.trainset = [X[0: training_size, :].astype(config.floatX, copy=False),
                         Y[0: training_size, :].astype(config.floatX, copy=False)]

        self.tuning = [X[training_size: training_size + tuning_size, :].astype(config.floatX, copy=False),
                       Y[training_size: training_size + tuning_size, :].astype(config.floatX, copy=False)]

        self.testset = [
            X[training_size + tuning_size: training_size + tuning_size + test_size, :].astype(config.floatX,
                                                                                                         copy=False),
            Y[training_size + tuning_size: training_size + tuning_size + test_size, :].astype(config.floatX,
                                                                                                         copy=False)]

        self.x_y_mapping = {
            'train': numpy.identity(self.trainset[0].shape[0]),
            'test': numpy.identity(self.testset[0].shape[0]),
            'dev': numpy.identity(self.tuning[0].shape[0])
        }

        self.x_reduce = {
            'train': range(self.trainset[0].shape[0]),
            'test': range(self.testset[0].shape[0]),
            'dev': range(self.tuning[0].shape[0])
        }

    def load_fish_database(self, database_path, fish_path):

        mat = scipy.io.loadmat(database_path)
        database = mat.get('database')[0]

        X = numpy.ndarray([database.shape[0], 11111])

        first_name = []
        last_name = []
        for index, record in enumerate(database):
            name = record[0][0]
            a = name.rfind(r'/') + 1
            b = name.find(r'+')
            c = b + 1
            d = name.find(r'_')

            first_name.append(name[a:b])
            last_name.append(name[c:d])

            X[index, :] = numpy.cast['float32'](database[index][6]).T

        f = open(fish_path, 'rb')

        f.seek(0, os.SEEK_END)
        endPos = f.tell()
        f.seek(0, os.SEEK_SET)

        dat = list(struct.unpack('{0}c'.format(endPos), f.read()))

        datA = dat[0: endPos - 1]
        datB = dat[1: endPos]

        line_starts = [i for i, val in enumerate(datA) if val == '>']
        line_ends = [i + 1 for i, (valA, valB) in enumerate(zip(datA, datB)) if
                     'z' >= valA >= 'a' and chr(10) <= valB <= chr(13)]

        minus_starts = [i + 1 for i, (valA, valB) in enumerate(zip(datA, datB)) if valA < ' ' and valB == '-']
        minus_ends = [i + 1 for i, (valA, valB) in enumerate(zip(datA, datB)) if valB < ' ' and valA == '-']

        n = len(line_starts)

        idx = numpy.ndarray(n, dtype=config.floatX)

        for j, (line_start, line_end) in enumerate(zip(line_starts, line_ends)):
            s = datA[line_start: line_end]
            f = [i for i, val in enumerate(s) if val == '|']
            sp = [i for i, val in enumerate(s) if val == ' ']

            fn = ''.join(s[f[-1] + 1: sp[0]])
            ln = ''.join(s[sp[-1] + 1::])

            idx[j] = self.gen_name_index(fn, ln, first_name, last_name)

        nuc_prob = numpy.ndarray([189, 901 * 4])

        for i in xrange(n):

            if idx[i] == -1:
                continue

            factor = 1. / sum(idx == idx[i])
            vec = numpy.array(dat[minus_starts[i]: minus_ends[i]])
            tacg = numpy.zeros([4, len(vec)])
            tacg[0, :] = (vec == 'T').astype(int)
            tacg[1, :] = (vec == 'A').astype(int)
            tacg[2, :] = (vec == 'C').astype(int)
            tacg[3, :] = (vec == 'G').astype(int)
            nuc_prob[idx[i], :] += (factor * tacg.flatten('F'))

        legal_rows = numpy.nonzero(numpy.sum(nuc_prob, axis=1))[0]

        first_name_legal = {}
        last_name_legal = {}

        for index, legal_col in enumerate(legal_rows):
            first_name_legal[index] = first_name[legal_col]
            last_name_legal[index] = last_name[legal_col]

        Y = nuc_prob[legal_rows, :]
        X = X[legal_rows, :]

        return X, Y

    def gen_name_index(self, first_name, last_name, fn_cell_arr, ln_cell_arr):

        idx = -1
        for i in xrange(189):
            if str(fn_cell_arr[i]).lower() == first_name.lower() and str(ln_cell_arr[i]).lower() == last_name.lower():
                return i

        return idx
