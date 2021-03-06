__author__ = 'aviv'

import sys

import numpy

from tester_base import TesterBase
from MISC.utils import cca_web2, calculate_mardia, calculate_trace, calculate_corrcoef


class CCACorraltionTester(TesterBase):

    def __init__(self, test_set_x, test_set_y, train_set_x, train_set_y, top=50):
        super(CCACorraltionTester, self).__init__(test_set_x, test_set_y)

        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        self.top = top

    def _calculate_metric(self, x, y, transformer, hyperparamsers):

        train_x_tilde, train_y_tilde = transformer.compute_outputs(self.train_set_x, self.train_set_y, hyperparamsers)

        wx, wy, r = cca_web2(train_x_tilde.T, train_y_tilde.T)

        x_tilde = numpy.dot(wx, x)
        y_tilde = numpy.dot(wy, y)

        x_var = numpy.var(x_tilde, axis=1)
        y_var = numpy.var(y_tilde, axis=1)

        x_mean = numpy.mean(x_tilde, axis=1)
        y_mean = numpy.mean(y_tilde, axis=1)

        print 'x variance: mean %f, var %f\n' % (numpy.mean(x_var), numpy.var(x_var))
        print 'y variance: mean %f, var %f\n' % (numpy.mean(y_var), numpy.var(y_var))

        print 'x mean: mean %f, var %f\n' % (numpy.mean(x_mean), numpy.var(x_mean))
        print 'y mean: mean %f, var %f\n' % (numpy.mean(y_mean), numpy.var(y_mean))

        sys.stdout.flush()

        print 'cca:\n'

        print 'trace = %f\n' % calculate_trace(x_tilde, y_tilde, self.top)
        sys.stdout.flush()

        print 'corrcoef = %f\n' % calculate_corrcoef(x_tilde, y_tilde, self.top)
        sys.stdout.flush()

        result = calculate_mardia(x_tilde, y_tilde, self.top)

        print 'mardia = %f\n' % result
        sys.stdout.flush()

        return result


