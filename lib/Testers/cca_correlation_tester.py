__author__ = 'aviv'

import numpy

from tester_base import TesterBase
from MISC.utils import unitnorm, center, cca_web2

class CCACorraltionTester(TesterBase):

    def __init__(self, test_set_x, test_set_y, train_set_x, train_set_y, dim=50):
        super(CCACorraltionTester, self).__init__(test_set_x, test_set_y)

        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        self.dim = dim

    def _find_correlation(self, x, y, transformer):

        train_x_tilde, train_y_tilde = transformer.compute_outputs(self.train_set_x, self.train_set_y)

        wx, wy, r = cca_web2(train_x_tilde.T, train_y_tilde.T)

        x_tilde = numpy.dot(x.T, wx)
        y_tilde = numpy.dot(y.T, wy)

        forward = unitnorm(center(x_tilde.T))
        backward = unitnorm(center(y_tilde.T))

        s = numpy.linalg.svd(numpy.dot(forward, backward.T), compute_uv=False)

        return numpy.sum(s[0: self.dim - 1])


