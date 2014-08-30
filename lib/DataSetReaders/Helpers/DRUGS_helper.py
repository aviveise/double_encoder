import math

import scipy.io
import numpy
from MISC import utils


__author__ = 'aviv'


if __name__ == '__main__':

    mat = scipy.io.loadmat('/home/aviv/Project/DoubleEncoder/DataSet/Drugs/side_effects_data.mat')

    fingerprints = mat.get('fingerprints')
    effect_ids = mat.get('effect_id')
    effect_name = mat.get('effect_name')

    effect_ids_number = [int(id[1:]) for id in effect_ids]

    max_size = math.ceil(math.log(max(effect_ids_number), 2))

    effects_ids_bin = numpy.ndarray([effect_ids.shape[0], max_size])

    for i in xrange(effect_ids.shape[0]):

        id_bin = utils.convertInt2Bitarray(effect_ids_number[i])
        id_bin_size = id_bin.shape[1]
        effects_ids_bin[i, 0: max_size - id_bin_size] = numpy.zeros([1, max_size - id_bin_size])
        effects_ids_bin[i, max_size - id_bin_size: max_size] = id_bin

    print fingerprints