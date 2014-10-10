__author__ = 'aviv'

import numpy

from tester_base import TesterBase
from MISC.utils import unitnorm, center, cca_web2

class CCACorraltionTester(TesterBase):

    def __init__(self, test_set_x, test_set_y, train_set_x, train_set_y, top=50):
        super(CCACorraltionTester, self).__init__(test_set_x, test_set_y)

        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        self.top = top

    def _find_correlation(self, x, y, transformer):

        train_x_tilde, train_y_tilde = transformer.compute_outputs(self.train_set_x, self.train_set_y)

        wx, wy, r = cca_web2(train_x_tilde.T, train_y_tilde.T)

        x_tilde = numpy.dot(wx, x)
        y_tilde = numpy.dot(wy, y)

        forward = unitnorm(center(x_tilde))
        backward = unitnorm(center(y_tilde))

        diag = numpy.abs(numpy.diagonal(numpy.dot(forward, backward.T)))
        diag.sort()
        diag = diag[::-1]

        #forward = center(x_tilde)
        #backward = center(y_tilde)

        #u_x, s_x, v_x = numpy.linalg.svd(forward)
        #u_y, s_y, v_y = numpy.linalg.svd(backward)

        #k = min(forward.shape[1], forward.shape[0])

        #temp_x = numpy.dot(u_x[:, 0:k], v_x[0:k, :])
        #temp_y = numpy.dot(u_y[:, 0:k], v_y[0:k, :])

        #corr = numpy.dot(temp_x, temp_y.T)

        #s = numpy.linalg.svd(corr, compute_uv=False)

        #return numpy.sum(s[0:50])

        return sum(diag[0:self.top])

