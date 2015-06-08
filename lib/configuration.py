import ConfigParser
import theano

from MISC.utils import ConfigSectionMap
from hyper_parameters import HyperParameters
import theano.tensor as Tensor
from softSigmoid import SoftSigmoid, soft_sigmoid

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

            if section.startswith('regularization_'):
                self.regularizations_parameters.append(map)

            elif section.startswith('strategy_'):
                self.strategy_parameters[map['name']] = map

        self.output_parameters = self._parse_output_parameters(ConfigSectionMap("output", config))
        self.hyper_parameters = self._parse_training_parameters(ConfigSectionMap("hyper_parameters", config))

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
        strategy = training_section['strategy']
        rho = float(training_section['rho'])
        cascade_train = bool(int(training_section['cascade_train']))

        return HyperParameters(layer_sizes=layer_sizes,
                               learning_rate=learning_rate,
                               batch_size=batch_size,
                               epochs=epochs,
                               momentum=momentum,
                               method_in=method_in,
                               method_out=method_out,
                               training_strategy=strategy,
                               rho=rho,
                               cascade_train=cascade_train)

    def convert_method(self, method_string):

        if method_string == 'sigmoid':
            return Tensor.nnet.sigmoid

        elif method_string == 'tanh':
            return Tensor.tanh

        elif method_string == 'scaled_tanh':
            return lambda x: 1.7159 * Tensor.tanh(0.66 * x)

        elif method_string == 'soft_sigmoid':
            return soft_sigmoid

        elif method_string == 'relu':
            return lambda x: x * (x > 0)

        elif method_string == 'relu2':
            return lambda x: Tensor.switch(Tensor.lt(x, -1), -1, x) * Tensor.switch(Tensor.gt(x, 1), 1, x) / x

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
            'import_net': output_section['import_net'],
            'verbosity': output_section['verbosity']
        }

        return output_params