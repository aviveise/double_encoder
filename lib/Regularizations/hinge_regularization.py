__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor


class HingeRegularization(RegularizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(HingeRegularization, self).__init__(regularization_parameters)

        self._k = float(regularization_parameters['k'])
        self._eps = float(regularization_parameters['eps'])
        self._b = float(regularization_parameters['b'])

    def compute(self, symmetric_double_encoder, params):

        regularization = 0
        #for layer in symmetric_double_encoder:
        hidden_x = symmetric_double_encoder[0].output_forward_x
        hidden_y = symmetric_double_encoder[-1].output_forward_y

            #hinge_x = self._threshold - (hidden_x / self._k) ** 2
            #hinge_y = self._threshold - (hidden_y / self._k) ** 2

            #hinge_x = 1 - Tensor.sqrt(abs(hidden_x / self._k))
            #hinge_y = 1 - Tensor.sqrt(abs(hidden_y / self._k))

        hinge_x = 1 - Tensor.log(hidden_x ** 2 / self._k + self._eps)
        hinge_y = 1 - Tensor.log(hidden_y ** 2 / self._k + self._eps)

            #hinge_x = 1 - Tensor.exp(-1 * abs(hidden_x) + 1) - self._k
            #hinge_y = 1 - Tensor.exp(-1 * abs(hidden_y) + 1) - self._k

            #hinge_x = 1 - Tensor.nnet.sigmoid(abs(hidden_x) / (2 * self._k))
            #hinge_y = 1 - Tensor.nnet.sigmoid(abs(hidden_y) / (2 * self._k))

            #(1 / self._b) * Tensor.log(1 + Tensor.exp(self._b * abs(hidden_x)))
            #(1 / self._b) * Tensor.log(1 + Tensor.exp(self._b * abs(hidden_y)))

        regularization += Tensor.sum(hinge_x * (hinge_x > 0))
        regularization += Tensor.sum(hinge_y * (hinge_y > 0))

        regularization -= self._zeroing_param

        return self.weight * regularization * (regularization > 0)

    def print_regularization(self, output_stream):
        super(HingeRegularization, self).print_regularization(output_stream)
