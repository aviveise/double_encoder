__author__ = 'aviv'

from regularization_base import RegularizationBase
from MISC.container import ContainerRegisterMetaClass
from theano import tensor as Tensor

class PairWiseCorrelationRegularization(RegularizationBase):

    __metaclass__ = ContainerRegisterMetaClass

    def __init__(self, regularization_parameters):
        super(PairWiseCorrelationRegularization, self).__init__(regularization_parameters)

        self.euc_length = bool(int(regularization_parameters['euc_length']))
        self.pair_wise = bool(int(regularization_parameters['pair_wise_correlation']))
        self.variance = bool(int(regularization_parameters['variance']))

    def compute(self, symmetric_double_encoder, params):

        regularization = 0;

        for layer in symmetric_double_encoder:

            forward = layer.output_forward.T
            backward = layer.output_backward.T

            mean_forward = Tensor.mean(forward, axis=1, dtype=Tensor.config.floatX)
            mean_backward = Tensor.mean(backward, axis=1, dtype=Tensor.config.floatX)

            forward_centered = forward - mean_forward.reshape([forward.shape[0], 1])
            backward_centered = backward - mean_backward.reshape([backward.shape[0], 1])

            if self.euc_length:
                regularization += Tensor.mean((forward_centered - backward_centered) ** 2)
                print 'added euc reg'

            if self.pair_wise:
                regularization += Tensor.mean((Tensor.dot(forward_centered, forward_centered.T) - Tensor.eye(forward.shape[0], dtype=Tensor.config.floatX)) ** 2)
                regularization += Tensor.mean((Tensor.dot(backward_centered, backward_centered.T) - Tensor.eye(backward.shape[0], dtype=Tensor.config.floatX)) ** 2)
                print 'added pair reg'

            if self.variance:
                regularization -= Tensor.mean(Tensor.dot(forward_centered, forward_centered.T) ** 2)
                regularization -= Tensor.mean(Tensor.dot(backward_centered, backward_centered.T) ** 2)
                print 'added var reg'


        return self.weight * regularization


    def print_regularization(self, output_stream):

        super(PairWiseCorrelationRegularization, self).print_regularization(output_stream)
