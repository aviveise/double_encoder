import os

import numpy
import scipy.io
from theano import config
from MISC import utils

from DataSetReaders.dataset_factory import dataset_meta
from DataSetReaders.dataset_base import DatasetBase

TRAINING_PERCENT = 0.8

class FISHDataSet(DatasetBase):

    __metaclass__ = dataset_meta

    def __init__(self, dataset_path, center, normalize, whiten):
        super(FISHDataSet, self).__init__(dataset_path, 'FISH', center, normalize, whiten)

    def build_dataset(self):

        X, Y = self.load_fish_database(self.dataset_path + 'fish_database.mat', self.dataset_path + 'fasta.fas')


        training_size = int(drug_number * TRAINING_PERCENT * 0.9)
        tuning_size = int(drug_number * TRAINING_PERCENT * 0.1)
        test_size = int(drug_number * (1 - TRAINING_PERCENT))

        self.trainset = [fingerprints[0: training_size, :].T.astype(config.floatX, copy=False),
                         side_effects[0: training_size, :].T.astype(config.floatX, copy=False)]

        self.tuning = [fingerprints[training_size: training_size + tuning_size, :].T.astype(config.floatX, copy=False),
                       side_effects[training_size: training_size + tuning_size, :].T.astype(config.floatX, copy=False)]

        self.testset = [fingerprints[training_size + tuning_size: training_size + tuning_size + test_size, :].T.astype(config.floatX, copy=False),
                        side_effects[training_size + tuning_size: training_size + tuning_size + test_size, :].T.astype(config.floatX, copy=False)]

    def load_fish_database(self, database_path, fish_path):

        mat = scipy.io.loadmat(database_path)
        database = mat.get('database')

        X = numpy.ndarray([11111, database.shape[0]])

        first_name = {}
        last_name = {}
        for i in xrange(database.shape[1]):

            s = list(database[1, i][0][0])
            a = max([i for (i, val) in s if val == '/']) + 1
            b = min([i for (i, val) in s if val == '+']) - 1
            c = b + 2
            d = min([i for (i, val) in s if val == '_']) - 1

            first_name[i] = s[a:b]
            last_name[i] = s[c:d]

            X[:, i] = database[1, i][6][0]

        f = open(fish_path, 'rb')

        f.seek(0, os.SEEK_END)
        endPos = f.tell()
        f.seek(0, os.SEEK_SET)

        data_size = endPos / numpy.finfo(numpy.uint8).nexp

        dat = list(f.readall())

        datA = dat[0: data_size - 1]
        datB = dat[1: data_size]

        line_starts = ([i for (i, val) in datA if val == '>'])
        line_ends_A = ([i for (i, val) in datA if val <= 'z' and val >= 'a'])
        line_ends_B = ([i for (i, val) in datB if val <= 13 and val >= 10])

        line_ends = ([x for x in line_ends_A for y in line_ends_B if x == y])

        minus_starts_A = ([i for (i, val) in datA if val < ''])
        minus_starts_B = ([i for (i, val) in datB if val == '-'])

        minus_starts = ([x for x in minus_starts_A for y in minus_starts_B if x == y]) + 1

        minus_ends_A = ([i for (i, val) in datA if val == '-'])
        minus_ends_B = ([i for (i, val) in datB if val < ' '])

        minus_ends = ([x for x in minus_ends_A for y in minus_ends_B if x == y]) + 1

        n = len(line_starts)

        idx = numpy.ndarray([n, 1], dtype=config.floatX)

        for i in xrange(n):
            s = datA[line_starts[i] : line_ends[i]]
            f = ([i for (i, val) in s if val == '|'])
            sp = ([i for (i, val) in s if val == ' '])

            fn = s[f[-1] + 1: s[0] - 1].T
            ln = s[sp[-1] + 1: -1].T

            idx[i] = self.gen_name_index(fn, ln, first_name, last_name)

        nuc_prob = numpy.ndarray([901 * 4, 189])

        for i in xrange(n):

            if idx[i] == -1:
                continue

            factor = 1 / sum([x for x in idx if x == idx[i]])
            vec = dat[minus_starts[i] : minus_ends[i]]
            tacg = numpy.ndarray([4, len(vec)])
            tacg[0, :] = [j for (j, val) in vec if (val == 'T')]
            tacg[1, :] = [j for (j, val) in vec if (val == 'A')]
            tacg[2, :] = [j for (j, val) in vec if (val == 'C')]
            tacg[3, :] = [j for (j, val) in vec if (val == 'G')]
            nuc_prob[:, idx[i]] = nuc_prob[:, idx[i]] + (factor * tacg[:])

        legal_cols = [i for (i, val) in sum(nuc_prob) if val > 0]

        first_name_legal = {}
        last_name_legal = {}

        for i in len(legal_cols):
            first_name_legal[i] = first_name[legal_cols[i]]
            last_name_legal[i] = last_name[legal_cols[i]]


        Y = nuc_prob[:, legal_cols]

        #Center X
        X = utils.center(X)
        Y = utils.center(Y)

        return X, Y

    def gen_name_index(self, first_name, last_name, fn_cell_arr, ln_cell_arr):

        idx = -1
        for i in xrange(189):
            if fn_cell_arr[i] == first_name:
                if ln_cell_arr[i] == last_name:
                    idx = i
                    return idx

        return idx