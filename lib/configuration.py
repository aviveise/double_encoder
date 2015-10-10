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
        validation_epoch = int(training_section['validation_epoch'])
        decay = self._listify(training_section['decay'])
        decay_factor = float(training_section['decay_factor'])
        early_stopping = bool(int(training_section['early_stopping']))
        early_stopping_layer = int(training_section['early_stopping_layer'])
        early_stopping_metric = training_section['early_stopping_metric']

        return HyperParameters(layer_sizes=layer_sizes,
                               learning_rate=learning_rate,
                               batch_size=batch_size,
                               epochs=epochs,
                               momentum=momentum,
                               method_in=method_in,
                               method_out=method_out,
                               training_strategy=strategy,
                               rho=rho,
                               cascade_train=cascade_train,
                               decay=decay,
                               decay_factor=decay_factor,
                               early_stopping=early_stopping,
                               early_stopping_layer=early_stopping_layer,
                               early_stopping_metric=early_stopping_metric,
                               validation_epoch=validation_epoch)

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

        elif method_string == 'leakyrelu':
            return lambda x: x * (x > 0) + 0.01 * x * (x < 0)

        elif method_string == 'shiftedrelu':
            return lambda x: x * (x > -1)

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
            'verbosity': output_section['verbosity'],
            'visualize': bool(int(output_section.get('visualize',0)))
        }

        return output_params

    def _listify(self, string):
        string = string.strip()

        if string == 'None':
            return None

        # Making sure we are truly working on a string the symbolizes a list.
        if string[0] != '[' or string[-1] != ']':
            raise Exception('Conversion to list failed of value {0}'.format(string))

        # Remove whitespaces
        string.replace(' ','')

        try:
            output_list = map(self._convert_value, string[1:-1].strip().split(','))
        except:
            raise Exception('Conversion to list failed of value {0}'.format(string))

        return output_list


    def _convert_value(self, string_value):
        for conversion in (self._listify, self._boolify, int, float):
            try:
                converted_value = conversion(string_value)
                return converted_value
            except:
                continue

        return string_value

    def _boolify(self, string):

        if string == 'True':
            return True
        elif string == 'False':
            return False
        else:
            raise Exception("Conversion of {0} to bool failed".format())
