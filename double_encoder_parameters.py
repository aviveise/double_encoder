import ConfigParser

from MISC.utils import ConfigSectionMap
from hyper_parameters import HyperParameters
from Optimizations import interval

__author__ = 'Aviv Eisenschtat'

class DoubleEncoderParameters(object):

    def __init__(self, config_file):

        config = ConfigParser.ConfigParser()
        config.read(config_file)

        data_section = ConfigSectionMap("Dataset", config)
        runtime_section = ConfigSectionMap("Runtime", config)
        training_section = ConfigSectionMap("Training", config)
        regularization_section = ConfigSectionMap("Regularization", config)
        structure_optimization_section = ConfigSectionMap("Structure Optimization" ,config)
        optimization_section = ConfigSectionMap("Optimization", config)

        self.base_hyper_parameters = HyperParameters()

        self.parse_training_parameters(training_section, regularization_section)
        self.parse_data_section(data_section)
        self.parse_runtime_section(runtime_section)
        self.parse_structure_section(structure_optimization_section)
        self.parse_optimization_section(optimization_section)

    def parse_runtime_section(self, runtime_section):
        self.gpu = bool(int(runtime_section['gpu']))
        self.optimization_mode = bool(int(runtime_section['optimization']))
        self.optimization_type = runtime_section['optimization_type']
        self.cca_type = runtime_section['cca_type']

    def parse_data_section(self, data_section):
        self.data_type = data_section['data_type']
        self.dataset_path = data_section['dataset_path']
        self.center = bool(int(data_section['center']))
        self.normalize = bool(int(data_section['normalize']))
        self.whiten = bool(int(data_section['whiten']))

    def parse_training_parameters(self, training_section, regularization_section):
        self.base_hyper_parameters.learning_rate = float(training_section['learning_rate'])
        self.base_hyper_parameters.batch_size = int(training_section['batch_size'])
        self.base_hyper_parameters.epochs = int(training_section['epochs'])
        self.base_hyper_parameters.momentum = float(training_section['momentum'])
        self.base_hyper_parameters.layer_sizes = map(int, training_section['layer_sizes'].split())
        self.base_hyper_parameters.regularization_type = training_section['regularization_type']

        self.base_hyper_parameters.regularization_parameters = {}

        for key, value in regularization_section.iteritems():
            self.base_hyper_parameters.regularization_parameters [key] = float(value)


    def parse_optimization_section(self, optimization_section):
        self.optimization_interval = interval.Interval(float(optimization_section['start_value']),
                                                       float(optimization_section['end_value']),
                                                       float(optimization_section['step']))

        self.optimization_priority = int(optimization_section['priority'])

    def parse_structure_section(self, structure_section):
        self.structure_optimization_layer_number = int(structure_section['hidden_layer_number'])
        self.structure_optimization_symmetric =  bool(int(structure_section['symmetric']))