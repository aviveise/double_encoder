import lasagne
from lib.MISC.logger import OutputLog
from lib.lasagne.Layers.LocallyDenseLayer import TiedDenseLayer, LocallyDenseLayer
from lib.lasagne.Layers.TiedNoiseLayer import TiedGaussianNoiseLayer
from lib.lasagne.Layers.TiedDropoutLayer import TiedDropoutLayer, DropoutLayer


class Params:
    # region Training Params
    BATCH_SIZE = 25
    VALIDATION_BATCH_SIZE = 1000
    CROSS_VALIDATION = True
    EPOCH_NUMBER = 60
    DECAY_EPOCH = [10, 20, 30, 40, 50, 60]
    DECAY_RATE = 0.5
    BASE_LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    # endregion

    # region Loss Weights
    WEIGHT_DECAY = 0.005
    GAMMA_COEF = 0.005
    WITHEN_REG_X = 0.5
    WITHEN_REG_Y = 0.5
    L2_LOSS = 0.25
    LOSS_X = 1
    LOSS_Y = 1
    SHRINKAGE = 0
    # endregion

    # region Architecture
    LAYER_SIZES = [600, 1200, 600]
    TEST_LAYER = 1
    DROPOUT = [0, 0.5, 0]
    RESCALE = True
    PARALLEL_WIDTH = 2
    WEIGHT_INIT = lasagne.init.GlorotUniform()
    LAYER_TYPES = [TiedDenseLayer, TiedDenseLayer, TiedDenseLayer, TiedDenseLayer]
    LEAKINESS = 0.3
    CELL_NUM = 4
    NOISE_LAYER = TiedDropoutLayer
    BN = True
    BN_ACTIVATION = False
    SIMILARITY_METRIC = 'correlation'
    # endregion

    @classmethod
    def print_params(cls):
        OutputLog().write('Params:\n')
        for (key, value) in cls.__dict__.iteritems():
            if not key.startswith('__'):
                OutputLog().write('{0}: {1}'.format(key, value))
