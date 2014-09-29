__author__ = 'aviv'

from theano import function
from double_encoder_tester import DoubleEncoderTester
from sklearn.cross_decomposition import CCA

class DoubleEncoderCCATester(DoubleEncoderTester):

    def __init__(self, double_encoder, layer_num, train_x, train_y, cca_dim=50):
        super(DoubleEncoderTester, self).__init__(double_encoder)

        self._layer_num = layer_num

        #Training inputs x1 and x2 as a matrices with columns as samples
        self._x = self._correlation_optimizer.var_x
        self._y = self._correlation_optimizer.var_y

        self.train_x = train_x
        self.train_y = train_y

        self.cca = CCA(cca_dim)

    def compute_outputs(self, test_set_x, test_set_y):

        model = self._build_model()

        train_x_tilde, train_y_tilde = model(self.train_x.T, self.train_y.T)

        self.cca.fit(train_x_tilde, train_y_tilde)

        test_x_tilde, test_y_tilde = model(test_set_x, test_set_y)

        return self.cca.transform(test_x_tilde, test_y_tilde)

