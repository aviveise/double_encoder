import numpy
import pycuda
from skcuda import cublas
import theano
from theano.sandbox.cuda import GpuOp, as_cuda_ndarray_variable, CudaNdarrayType
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda.fftconv import bptrs, scikits

__author__ = 'avive'


class BatchedInvOp(GpuOp):
    __props__ = ()

    def make_node(self, input):
        input = gpu_contiguous(as_cuda_ndarray_variable(input))

        self.destructive = True

        assert input.dtype == "float32"
        assert input.ndim == 3  # (batch, a, b)

        return theano.Apply(self, [input],
                            [self.output_type(input)()])

    def output_type(self, input):
        return CudaNdarrayType((input.type.broadcastable[0], input.type.broadcastable[2]))

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]
        from theano.misc.pycuda_utils import to_gpuarray

        # reusable allocations
        pivot_alloc = [None]
        info_alloc = [None]

        def thunk():
            input_shape = inputs[0][0].shape

            size = input_shape[1]  # matrices to invert are (size x size)
            batch_size = input_shape[0]

            z = outputs[0]

            # only allocate if there is no previous allocation of the right size.
            if z[0] is None or z[0].shape != input_shape:
                z[0] = theano.sandbox.cuda.CudaNdarray.zeros(input_shape)
                pivot_alloc[0] = pycuda.gpuarray.empty((batch_size, size), numpy.int32)
                info_alloc[0] = pycuda.gpuarray.zeros(batch_size, numpy.int32)

            input_pycuda = to_gpuarray(inputs[0][0])
            output_pycuda = to_gpuarray(z[0])
            pivot = pivot_alloc[0]
            info = info_alloc[0]

            if not self.destructive:
                input_pycuda = input_pycuda.copy()  # to prevent destruction of the input

            # construct pointer arrays for batched operations
            input_arr = bptrs(input_pycuda)
            output_arr = bptrs(output_pycuda)

            handle = scikits.cuda.misc._global_cublas_handle

            # perform LU factorization
            cublas.cublasSgetrfBatched(handle, size, input_arr.gpudata, size, pivot.gpudata, info.gpudata, batch_size)
            # the LU factorization is now in input_pycuda (destructive operation!)

            # use factorization to perform inversion
            cublas.cublasSgetriBatched(handle, size, input_arr.gpudata, size, pivot.gpudata, output_arr.gpudata, size,
                                       info.gpudata, batch_size)
            # the inverted matrices are now in output_pycuda

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk


batched_inv = BatchedInvOp()

