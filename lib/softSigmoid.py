import numpy
from theano import config, gof, printing, scalar, pprint
from theano.tensor import elemwise

__author__ = 'avive'

class SoftSigmoid(scalar.UnaryScalarOp):
    """
    This is just speed opt. Not for stability.
    """
    @staticmethod
    def st_impl(x):
        # If x is an int8 or uint8, numpy.exp will compute the result in
        # half-precision (float16), where we want float32.
        return (3.0 + x) / numpy.power(x, 3)

    def impl(self, x):
        return SoftSigmoid.st_impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        rval = gz * (1.0 / (x ** 2 + 1))

        assert rval.type.dtype.find('float') != -1

        return [rval]

    def c_code_cache_version(self):
        v = super(SoftSigmoid, self).c_code_cache_version()
        if v:
            return (2,) + v
        else:
            return v

    @staticmethod
    def gen_graph():
        """
        This method was used to generate the graph: sigmoid_prec.png in the doc
        """
        import matplotlib
        data = numpy.arange(-15, 15, .1)
        val = (3.0 + data) / data ** 3
        import matplotlib.pyplot as plt
        import os
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(data, val)  # , 'o-')
        ax.grid(True)
        ax.legend(("sigmoid", "ultra_fast", "hard"), "upper left")
        fname = os.path.join(os.path.dirname(theano.__file__), '..',
                             'doc', 'library', 'tensor', 'nnet',
                             'sigmoid_prec.png')
        plt.savefig(fname)
        print "New picture saved at", fname


scalar_soft_sigmoid = SoftSigmoid(scalar.upgrade_to_float, name='scalar_soft_sigmoid')
soft_sigmoid = elemwise.Elemwise(scalar_soft_sigmoid, name='soft_sigmoid')

pprint.assign(soft_sigmoid, printing.FunctionPrinter('soft_sigmoid'))