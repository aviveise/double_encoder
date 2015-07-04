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
        self._hinge_strategy = regularization_parameters['hinge_strategy']
        self._hinge_type = regularization_parameters['hinge_type']

    def compute(self, symmetric_double_encoder, params):

        regularization = 0
        if self._hinge_strategy == 'twin':
            hidden_x = symmetric_double_encoder[0].output_forward_y
            hidden_y = symmetric_double_encoder[-1].output_forward_x

            regularization += self.calc_hinge(hidden_x)
            regularization += self.calc_hinge(hidden_y)

        elif self._hinge_strategy == 'all':
            for layer in symmetric_double_encoder:
                hidden_x = layer.output_forward_y
                hidden_y = layer.output_forward_x

                regularization += self.calc_hinge(hidden_x)
                regularization += self.calc_hinge(hidden_y)

        return self.weight * regularization

    def print_regularization(self, output_stream):
        super(HingeRegularization, self).print_regularization(output_stream)

    def calc_hinge(self, x):

        if self._hinge_type == 'log':
            hinge = Tensor.sum(1 - Tensor.log(x ** 2 * self._k + self._b))

        elif self._hinge_type == 'abs':
            hinge = 1 - abs(x * self._k)

        elif self._hinge_type == 'sqr':
            hinge = Tensor.sum(1 - x ** 2 * self._k)

        return Tensor.sum(hinge * (hinge > 0))
