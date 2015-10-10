from MISC.logger import OutputLog
import os
import pickle

__author__ = 'aviv'

import abc

from tabulate import tabulate

class TesterBase(object):

    def __init__(self, test_set_x, test_set_y, visualize=False):
        self._x = test_set_x
        self._y = test_set_y
        self._visualize = visualize
        self._reduce_x = 0

        self._metrics = {}

    def test(self, transformer, hyperparamsers):

        hidden_values = transformer.compute_outputs(self._x, self._y, hyperparamsers)

        #Printing correlation scores for each hidden layer
        correlation = 0
        zipped = zip(hidden_values[0], hidden_values[1])
        index = 0

        table_header = ['layer']
        table_header.extend(self._headers())

        table_rows = []

        outputs_x = []
        outputs_y = []
        layer_id = 0
        lowest_var = 1

        correlations = []

        for x_hid, y_hid in zipped:

            row_hidden = ["layer {0} - hidden".format(index)]

            #calculation correlation between hidden values
            correlation_temp_hidden, metrics = self._calculate_metric(x_hid,
                                                                      y_hid,
                                                                      transformer,
                                                                      row_hidden)

            correlations.append(correlation_temp_hidden)

            table_rows.append(row_hidden)

            outputs_x.append(x_hid)
            outputs_y.append(y_hid)

            if correlation_temp_hidden > correlation:
                correlation = correlation_temp_hidden
                layer_id = index

            for metric in metrics.keys():
                if index not in self._metrics:
                    self._metrics[index] = {}

                if metric not in self._metrics[index]:
                    self._metrics[index][metric] = []

                self._metrics[index][metric].append(metrics[metric])

            index += 1

        OutputLog().write(tabulate(table_rows, headers=table_header))

        return correlations, correlation, lowest_var, outputs_x, outputs_y, layer_id

    @abc.abstractmethod
    def _calculate_metric(self, x, y, transformer, print_row):
        return

    @abc.abstractmethod
    def _headers(self):
        return

    def saveResults(self, path):
        pickle.dump(self._metrics,file(os.path.join(path, 'metrics.pkl'),'w'))
