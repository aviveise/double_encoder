__author__ = 'aviv'

from MISC.utils import print_list

class DccaHyperParameters(object):

    def __init__(self, LayerWidthH=[0 ,0],
                       ccaReg1=0,
                       ccaReg2=0,
                       L2H=[0, 0],
                       L2I=[0, 0],
                       L2O=[0, 0],
                       gaussianStdDevI=[0, 0],
                       gaussianStdDevH=[0, 0],
                       backpropReg=0):

        self.LayerWidthH = LayerWidthH
        self.ccaReg1 = ccaReg1
        self.ccaReg2 = ccaReg2
        self.L2H = L2H
        self.L2I = L2I
        self.L2O = L2O
        self.gaussianStdDevI = gaussianStdDevI
        self.gaussianStdDevH = gaussianStdDevH
        self.backpropReg=backpropReg

