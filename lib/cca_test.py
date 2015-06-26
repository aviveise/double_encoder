import ConfigParser
import os
import sys

from sklearn.cross_decomposition import CCA
from MISC.container import Container
from MISC.logger import OutputLog
from MISC.utils import ConfigSectionMap
from configuration import Configuration
from lib.Testers.trace_correlation_tester import TraceCorrelationTester
from lib.Transformers.identity_transform import IdentityTransformer

import DataSetReaders


__author__ = 'avive'

if __name__ == '__main__':

    data_set_config = sys.argv[1]
    run_time_config = sys.argv[2]
    top = int(sys.argv[3])

    regularization_methods = {}

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

    cca_model = CCA(n_components=top, scale=True, copy=False)

    train_transformed_x, train_transformed_y = cca_model.fit_transform(data_set.trainset[0], data_set.trainset[1])
    test_transformed_x, test_transformed_y = cca_model.transform(data_set.testset[0], data_set.testset[1])

    OutputLog().write('test results:')
    correlations, trace_correlation, var, x_test, y_test, test_best_layer = TraceCorrelationTester(
        data_set.testset[0],
        data_set.testset[1], top).test(IdentityTransformer(), configuration.hyper_parameters)

    OutputLog().write('train results:')
    correlations, train_trace_correlation, var, x_train, y_train, train_best_layer = TraceCorrelationTester(
        data_set.trainset[0],
        data_set.trainset[1], top).test(IdentityTransformer(), configuration.hyper_parameters)

    OutputLog().write('\nTest results : \n')

    configuration.hyper_parameters.print_parameters(OutputLog())

    OutputLog().write('\nResults:\n')

    OutputLog().write('trace: correlation execution_time\n')

    OutputLog().write('%f\n' % (float(trace_correlation)))


