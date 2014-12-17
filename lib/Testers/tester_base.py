__author__ = 'aviv'

import abc

from tabulate import tabulate

class TesterBase(object):

    def __init__(self, test_set_x, test_set_y):
        self._x = test_set_x
        self._y = test_set_y

    def test(self, transformer):

        hidden_values, output_values = transformer.compute_outputs(self._x, self._y)

        #Printing correlation scores for each hidden layer
        correlation = 0
        zipped = zip(hidden_values[0], hidden_values[1], output_values[0], output_values[1])
        index = 0

        table_header = ['layer']
        table_header.extend(self._headers())

        table_rows = []

        for x_hid, y_hid, x_out, y_out in zipped:

            row_hidden = ["layer {0} - hidden".format(index)]
            row_recons = ["layer {0} - hidden".format(index)]

            index += 1

            #calculation correlation between hidden values
            correlation_temp_hidden = self._calculate_metric(x_hid.T, y_hid.T, transformer, row_hidden)
            correlation_temp_reconstruct = self._calculate_metric(x_out.T, y_out.T, transformer, row_recons)

            table_rows.append(row_hidden)
            table_rows.append(row_recons)

            correlation_temp = max(correlation_temp_hidden,
                                   correlation_temp_reconstruct)

            if correlation_temp > correlation:
                correlation = correlation_temp


        print tabulate(table_rows, headers=table_header)
        print '\n'

        return correlation

    @abc.abstractmethod
    def _calculate_metric(self, x, y, transformer, print_row):
        return

    @abc.abstractmethod
    def _headers(self):
        return