import lasagne

from lib.MISC.logger import OutputLog
from lib.lasagne.Layers.LocallyDenseLayer import TiedDenseLayer, LocallyDenseLayer
from lib.lasagne.Layers.TiedNoiseLayer import TiedGaussianNoiseLayer
from lib.lasagne.Layers.TiedDropoutLayer import TiedDropoutLayer


class Params:

    # region Training Params
    BATCH_SIZE = 128
    EPOCH_NUMBER = 80
    DECAY_EPOCH = [10, 30, 50, 70]
    DECAY_RATE = 0.1
    BASE_LEARNING_RATE = 0.0001
    MOMENTUM = 0.9
    # endregion

    # region Loss Weights
    WEIGHT_DECAY = 0.005
    WITHEN_REG_X = 0.1
    WITHEN_REG_Y = 0.1
    L2_LOSS = 0.25
    LOSS_X = 1
    LOSS_Y = 1
    SHRINKAGE = 0.1
    # endregion

    # region Architecture
    LAYER_SIZES = [2000, 3000, 4000]
    TEST_LAYER = 1
    DROPOUT = [0.5, 0.5, 0.5]
    PARALLEL_WIDTH = 2
    WEIGHT_INIT = lasagne.init.GlorotUniform()
    LAYER_TYPES = [TiedDenseLayer, TiedDenseLayer, TiedDenseLayer, TiedDenseLayer]
    LEAKINESS = 0.3
    CELL_NUM = 2
    NOISE_LAYER = TiedDropoutLayer
    # endregion

    @classmethod
    def print_params(cls):
        OutputLog().write('Params:\n')
        for (key, value) in cls.__dict__.iteritems():
            if not key.startswith('__'):
                OutputLog().write('{0}: {1}'.format(key, value))
