__author__ = 'aviv'

import abc

from tabulate import tabulate

class TesterBase(object):

    def __init__(self, test_set_x, test_set_y):
        self._x = test_set_x
        self._y = test_set_y

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

        for x_hid, y_hid in zipped:

            row_hidden = ["layer {0} - hidden".format(index)]

            #calculation correlation between hidden values
            correlation_temp_hidden = self._calculate_metric(x_hid, y_hid, transformer, row_hidden)

            table_rows.append(row_hidden)

            outputs_x.append(x_hid)
            outputs_y.append(y_hid)

            if correlation_temp_hidden > correlation:
                correlation = correlation_temp_hidden
                layer_id = index

            index += 1

        print tabulate(table_rows, headers=table_header)
        print '\n'

        return correlation, lowest_var, outputs_x, outputs_y, layer_id

    @abc.abstractmethod
    def _calculate_metric(self, x, y, transformer, print_row):
        return

    @abc.abstractmethod
    def _headers(self):
        return