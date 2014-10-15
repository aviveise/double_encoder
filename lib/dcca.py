__author__ = 'aviv'

import sys
import ConfigParser
import DataSetReaders
import numpy

from Testers.trace_correlation_tester import TraceCorrelationTester
from Transformers.identity_transform import IdentityTransformer

from configuration import Configuration

from MISC.container import Container
from MISC.utils import ConfigSectionMap

__author__ = 'aviv'

if __name__ == '__main__':


    data_set_config = sys.argv[1]
    dcca_location = sys.argv[2]
    run_time_config = sys.argv[3]
    layer_number1 = int(sys.argv[4])
    layer_number2 = int(sys.argv[5])
    inFeat1 = int(sys.argv[6])
    inFeat2 = int(sys.argv[7])
    output_size = int(sys.argv[8])

    sys.path.append(dcca_location)

    import DccaWrapper

    configuration = Configuration(run_time_config)

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    #construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)

    dcca_hyper_parameters = configuration.dcca_hyper_parameters

    hyper_parameters = DccaWrapper.DCCAHyperParams()

    hyper_parameters.backpropReg = dcca_hyper_parameters.backpropReg
    hyper_parameters.ccaReg1= dcca_hyper_parameters.ccaReg1
    hyper_parameters.ccaReg1 = dcca_hyper_parameters.ccaReg1

    hyper_parameters.params[0].layerWidthH = dcca_hyper_parameters.layerWidthH[0]
    hyper_parameters.params[1].layerWidthH = dcca_hyper_parameters.layerWidthH[1]

    hyper_parameters.params[0].pretrainL2i = dcca_hyper_parameters.L2I[0]
    hyper_parameters.params[1].pretrainL2i = dcca_hyper_parameters.L2I[1]

    hyper_parameters.params[0].pretrainL2h = dcca_hyper_parameters.L2H[0]
    hyper_parameters.params[1].pretrainL2h = dcca_hyper_parameters.L2H[1]

    hyper_parameters.params[0].pretrainL2o = dcca_hyper_parameters.L2O[0]
    hyper_parameters.params[1].pretrainL2o = dcca_hyper_parameters.L2O[1]

    hyper_parameters.params[0].gaussianStdDevI = dcca_hyper_parameters.gaussianStdDevI[0]
    hyper_parameters.params[1].gaussianStdDevI = dcca_hyper_parameters.gaussianStdDevI[1]

    hyper_parameters.params[0].gaussianStdDevH = dcca_hyper_parameters.gaussianStdDevH[0]
    hyper_parameters.params[1].gaussianStdDevH = dcca_hyper_parameters.gaussianStdDevH[1]

    model = DccaWrapper.DeepCCAModel(hyper_parameters)

    train_modifiers = DccaWrapper.TrainMofidiers()

    train_modifiers.LBFGS_tol = 0.0001
    train_modifiers.LBFGS_M = 15
    train_modifiers.testGrad = False

    pretrain_train_modifiers = DccaWrapper.TrainMofidiers()

    pretrain_train_modifiers.LBFGS_tol = 0.001
    pretrain_train_modifiers.LBFGS_M = 15
    pretrain_train_modifiers.testGrad = False

    random = DccaWrapper.Random(-1)

    model.Train(hyper_parameters,
                [layer_number1, layer_number2],
                [inFeat1, inFeat2],
                output_size,
                train_data,
                pretrain_train_modifiers,
                train_modifiers,
                random)