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
        x_best = 0
        y_best = 0
        zipped = zip(hidden_values[0], hidden_values[1])
        index = 0

        table_header = ['layer']
        table_header.extend(self._headers())

        table_rows = []

        for x_hid, y_hid in zipped:

            row_hidden = ["layer {0} - hidden".format(index)]

            index += 1

            #calculation correlation between hidden values
            correlation_temp_hidden = self._calculate_metric(x_hid.T, y_hid.T, transformer, row_hidden)

            table_rows.append(row_hidden)

            if correlation_temp_hidden > correlation:
                correlation = correlation_temp_hidden
                x_best = x_hid
                y_best = y_hid

        zipped = zip(output_values[0], output_values[1])

        index = 0

        for x_rec, y_rec in zipped:

            row_recons = ["layer {0} - recons".format(index / 2)]

            index += 1

            correlation_temp_reconstruct = self._calculate_metric(x_rec.T, y_rec.T, transformer, row_recons)

            table_rows.append(row_recons)

            if correlation_temp_reconstruct > correlation:
                correlation = correlation_temp_reconstruct
                x_best = x_rec
                y_best = y_rec


        print tabulate(table_rows, headers=table_header)
        print '\n'

        return correlation, x_best, y_best

    @abc.abstractmethod
    def _calculate_metric(self, x, y, transformer, print_row):
        return

    @abc.abstractmethod
    def _headers(self):
        return