import ConfigParser

from MISC.utils import ConfigSectionMap
from hyper_parameters import HyperParameters

__author__ = 'Aviv Eisenschtat'

class Configuration(object):

    def __init__(self, config_file_path):

        config = ConfigParser.ConfigParser()
        config.read(config_file_path)

        self.optimizations_parameters = []
        self.regularizations_parameters = []

        sections = config.sections()
        for section in sections:

            if section.startswith('optimization_'):
                self.optimizations_parameters.append(ConfigSectionMap(section, config))

            elif section.startswith('regularization_'):
                self.regularizations_parameters.append(ConfigSectionMap(section, config))

        self.hyper_parameters = self._parse_training_parameters(ConfigSectionMap("hyper_parameters", config))

    def _parse_training_parameters(self, training_section):
        learning_rate = float(training_section['learning_rate'])
        batch_size = float(training_section['batch_size'])
        epochs = int(training_section['epochs'])
        momentum = float(training_section['momentum'])
        layer_sizes = map(int, training_section['layer_sizes'].split())

        return HyperParameters(layer_sizes=layer_sizes,
                               learning_rate=learning_rate,
                               batch_size=batch_size,
                               epochs=epochs,
                               momentum=momentum)