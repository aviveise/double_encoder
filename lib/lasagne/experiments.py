import numpy

from lib.lasagne.Layers.TiedDropoutLayer import TiedDropoutLayer, DropoutLayer

experiments = [
    [[('BN', True), ('GAMMA_COEF', 0.05), ('BN_ACTIVATION', False)],
     [('BN', True), ('GAMMA_COEF', 0.05), ('BN_ACTIVATION', True)],
     [('BN', False)],
     [('BN', True), ('GAMMA_COEF', 0), ('BN_ACTIVATION', False)],
     [('BN', True), ('GAMMA_COEF', 0), ('BN_ACTIVATION', True)]],

    [[('NOISE_LAYER', TiedDropoutLayer), ('DROPOUT', [0.5, 0.5, 0.5])],
     [('NOISE_LAYER', DropoutLayer), ('DROPOUT', [0.5, 0.5, 0.5])],
     [('NOISE_LAYER', TiedDropoutLayer), ('DROPOUT', [0, 0.5, 0])],
     [('NOISE_LAYER', DropoutLayer), ('DROPOUT', [0, 0.5, 0])],
     [('NOISE_LAYER', TiedDropoutLayer), ('DROPOUT', [0.5, 0, 0.5])],
     [('NOISE_LAYER', DropoutLayer), ('DROPOUT', [0.5, 0, 0.5])],
     [('NOISE_LAYER', DropoutLayer), ('DROPOUT', [0, 0, 0])]],

    [('LEAKINESS', numpy.arange(0, 1.1, 0.1))]
    [('GAMMA_COEF', numpy.arange(0, 0.6, 0.05))]
]
