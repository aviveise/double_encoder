import ConfigParser
import os
import sys
from MISC.container import Container
from MISC.logger import OutputLog
from MISC.utils import ConfigSectionMap, calculate_mardia
from lib.configuration import Configuration
from lib.DoubleEncoderCCA.DoubleEncoderCCA import DoubleEncoderTransform

import DataSetReaders

__author__ = 'avive'


if __name__ == '__main__':

    data_set_config = sys.argv[1]
    run_time_config = sys.argv[2]
    top = int(sys.argv[3])

    # parse runtime configuration
    configuration = Configuration(run_time_config)

    dir_name = configuration.output_parameters['path']

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    OutputLog().set_path(dir_name)
    OutputLog().set_verbosity(configuration.output_parameters['verbosity'])

    data_config = ConfigParser.ConfigParser()
    data_config.read(data_set_config)
    data_parameters = ConfigSectionMap("dataset_parameters", data_config)

    # construct data set
    data_set = Container().create(data_parameters['name'], data_parameters)

    t = DoubleEncoderTransform(configuration.output_parameters['import_net'],
                               r'C:\Theses\double_encoder\Datasets\XRMB\scaler_x.p',
                               r'C:\Theses\double_encoder\Datasets\XRMB\scaler_y.p')

    transformed_x = t.transform_x(data_set.testset[0])
    transformed_y = t.transform_y(data_set.testset[1])

    corr = calculate_mardia(transformed_x, transformed_y, top, False)

    print 'Correlation: {0}'.format(corr)