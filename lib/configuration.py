import ConfigParser

from MISC.utils import ConfigSectionMap
from hyper_parameters import HyperParameters
import theano.tensor as Tensor

__author__ = 'Aviv Eisenschtat'

class Configuration(object):

    def __init__(self, config_file_path):

        config = ConfigParser.ConfigParser()
        config.read(config_file_path)

        self.optimizations_parameters = []
        self.regularizations_parameters = []
        self.strategy_parameters = {}

        sections = config.sections()
        for section in sections:

            map = ConfigSectionMap(section, config)

            if section.startswith('optimization_'):
                self.optimizations_parameters.append(map)

            elif section.startswith('regularization_'):
                self.regularizations_parameters.append(map)

            elif section.startswith('strategy_'):
                self.strategy_parameters[map['name']] = map

        self.output_parameters = self._parse_output_parameters(ConfigSectionMap("output", config))
        self.hyper_parameters = self._parse_training_parameters(ConfigSectionMap("hyper_parameters", config))
        self.dcca_hyper_parameters = self._parse_dcca_training_parameters(ConfigSectionMap("dcca_hyper_parameters", config))

    def _parse_training_parameters(self, training_section):

        if training_section is None:
            return

        learning_rate = float(training_section['learning_rate'])
        batch_size = float(training_section['batch_size'])
        epochs = int(training_section['epochs'])
        momentum = float(training_section['momentum'])
        layer_sizes = map(int, training_section['layer_sizes'].split())
        method_in = self.convert_method(training_section['method_in'])
        method_out = self.convert_method(training_section['method_out'])
        output_layer_size = int(training_section['output_layer_size'])
        reg1 = float(training_section['reg1'])
        reg2 = float(training_section['reg2'])

        return HyperParameters(layer_sizes=layer_sizes,
                               learning_rate=learning_rate,
                               batch_size=batch_size,
                               epochs=epochs,
                               momentum=momentum,
                               method_in=method_in,
                               method_out=method_out,
                               output_layer_size=output_layer_size,
                               reg1=reg1,
                               reg2=reg2)

    def _parse_dcca_training_parameters(self, training_section):

        if training_section is None:
            return

        backpropReg = float(training_section['backpropReg'])

        ccaReg1 = float(training_section['ccaReg1'])
        ccaReg2 = float(training_section['ccaReg2'])

        L2H_side1 = float(training_section['L2H_side1'])
        L2I_side1 = float(training_section['L2I_side1'])
        L2O_side1 = float(training_section['L2O_side1'])
        L2H_side2 = float(training_section['L2H_side2'])
        L2I_side2 = float(training_section['L2I_side2'])
        L2O_side2 = float(training_section['L2O_side2'])

        LayerWidthH_side1= float(training_section['LayerWidthH_side1'])
        LayerWidthH_side2= float(training_section['LayerWidthH_side2'])

        gaussianStdDevI_side1= float(training_section['gaussianStdDevI_side1'])
        gaussianStdDevH_side1= float(training_section['gaussianStdDevH_side1'])
        gaussianStdDevI_side2= float(training_section['gaussianStdDevI_side2'])
        gaussianStdDevH_side2= float(training_section['gaussianStdDevH_side2'])

        return DccaHyperParameters(LayerWidthH = [LayerWidthH_side1 ,LayerWidthH_side2],
                                   ccaReg1=ccaReg1,
                                   ccaReg2=ccaReg2,
                                   L2H=[L2H_side1, L2H_side2],
                                   L2I=[L2I_side1, L2I_side2],
                                   L2O=[L2O_side1, L2O_side2],
                                   gaussianStdDevI=[gaussianStdDevI_side1, gaussianStdDevI_side2],
                                   gaussianStdDevH=[gaussianStdDevH_side1, gaussianStdDevH_side2],
                                   backpropReg=backpropReg)

    def convert_method(self, method_string):

        if method_string == 'sigmoid':
            return Tensor.nnet.sigmoid

        elif method_string == 'relu':
            return lambda x: x * (x > 0)

        elif method_string == 'hard_sigmoid':
            return Tensor.nnet.hard_sigmoid

        elif method_string == 'none':
            return lambda x: x

        else:
            raise Exception('method unknown')

    def _parse_output_parameters(self, output_section):

        output_params = {
            'path': output_section['path'],
            'type': output_section['type'],
            'sample': bool(int(output_section['sample'])),
            'sample_number': int(output_section['sample_number']),
            'fine_tune': bool(int(output_section['fine_tune'])),
            'import_net': output_section['import_net']
        }

        return output_params