__author__ = 'aviv'

import numpy

from tester_base import TesterBase
from sklearn.cross_decomposition import CCA
from MISC.utils import unitnorm, center

class CCACorraltionTester(TesterBase):

    def __init__(self, test_set_x, test_set_y, train_set_x, train_set_y, dim=50):
        super(CCACorraltionTester, self).__init__(test_set_x, test_set_y)

        self.cca = CCA(n_components=dim, max_iter=1000)
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y

    def _find_correlation(self, x, y, transformer):

        train_x_tilde, train_y_tilde = transformer.compute_outputs(self.train_set_x, self.train_set_y)

        print train_x_tilde
        print train_y_tilde

        self.cca.fit(train_x_tilde, train_y_tilde)
        x_tilde, y_tilde = self.cca.transform(x, y)

        forward = unitnorm(center(x_tilde))
        backward = unitnorm(center(y_tilde))

        s = numpy.linalg.svd(numpy.dot(forward, backward.T), compute_uv=False)

        return numpy.sum(s)


