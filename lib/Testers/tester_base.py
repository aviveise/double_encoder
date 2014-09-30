__author__ = 'aviv'

import abc

class TesterBase(object):

    def __init__(self, test_set_x, test_set_y):
        self._x = test_set_x
        self._y = test_set_y

    def test(self, transformer):
        x_tilde, y_tilde = transformer.compute_outputs(self._x, self._y)
        return self._find_correlation(x_tilde.T, y_tilde.T, transformer)

    @abc.abstractmethod
    def _find_correlation(self, x, y, transformer):
        return