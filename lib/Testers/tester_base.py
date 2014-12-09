__author__ = 'aviv'

import abc

class TesterBase(object):

    def __init__(self, test_set_x, test_set_y):
        self._x = test_set_x
        self._y = test_set_y

    def test(self, transformer):

        x_tilde, y_tilde = transformer.compute_outputs(self._x, self._y)
        correlation = 0
        zipped = zip(x_tilde, y_tilde)
        index = 0

        for x, y in zipped:

            print 'result - layer %d' % index
            index += 1

            correlation_temp = self._find_correlation(x.T, y.T, transformer)
            if correlation_temp > correlation:
                correlation = correlation_temp

        return correlation

    @abc.abstractmethod
    def _find_correlation(self, x, y, transformer):
        return