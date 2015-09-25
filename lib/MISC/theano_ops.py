import theano
from theano.sandbox.cuda import GpuOp, as_cuda_ndarray_variable, CudaNdarrayType
from theano.sandbox.cuda.basic_ops import gpu_contiguous

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
        return CudaNdarrayType((input.type.broadcastable[0], input.type.broadcastable[1], input.type.broadcastable[2]))

    # def make_thunk(self, node, storage_map, _, _2):
    #     inputs = [storage_map[v] for v in node.inputs]
    #     outputs = [storage_map[v] for v in node.outputs]
    #     from theano.misc.pycuda_utils import to_gpuarray
    #
    #     # reusable allocations
    #     pivot_alloc = [None]
    #     info_alloc = [None]
    #
    #     def thunk():
    #
    #         start = time()
    #
    #         input_shape = inputs[0][0].shape
    #
    #         size = input_shape[1]  # matrices to invert are (size x size)
    #         batch_size = input_shape[0]
    #
    #         z = outputs[0]
    #
    #         # only allocate if there is no previous allocation of the right size.
    #         if z[0] is None or z[0].shape != input_shape:
    #             z[0] = theano.sandbox.cuda.CudaNdarray.zeros(input_shape)
    #             pivot_alloc[0] = pycuda.gpuarray.empty((batch_size, size), numpy.int32)
    #             info_alloc[0] = pycuda.gpuarray.zeros(batch_size, numpy.int32)
    #
    #         input_pycuda = to_gpuarray(inputs[0][0])
    #         output_pycuda = to_gpuarray(z[0])
    #         pivot = pivot_alloc[0]
    #         info = info_alloc[0]
    #
    #         init = time()
    #
    #         print('init time:{0}'.format(init - start))
    #
    #         if not self.destructive:
    #             input_pycuda = input_pycuda.copy()  # to prevent destruction of the input
    #
    #         # construct pointer arrays for batched operations
    #         input_arr = bptrs(input_pycuda)
    #         output_arr = bptrs(output_pycuda)
    #
    #         alloc = time()
    #
    #         print('allocation time:{0}'.format(alloc - init))
    #
    #         handle = scikits.cuda.misc._global_cublas_handle
    #
    #         # perform LU factorization
    #         cublas.cublasSgetrfBatched(handle, size, input_arr.gpudata, size, pivot.gpudata, info.gpudata, batch_size)
    #         # the LU factorization is now in input_pycuda (destructive operation!)
    #
    #         LU = time()
    #
    #         print('LU time:{0}'.format(LU - alloc))
    #
    #         # use factorization to perform inversion
    #         cublas.cublasSgetriBatched(handle, size, input_arr.gpudata, size, pivot.gpudata, output_arr.gpudata, size,
    #                                    info.gpudata, batch_size)
    #         # the inverted matrices are now in output_pycuda
    #
    #         inv = time()
    #
    #         print('inv time:{0}'.format(inv - LU))
    #
    #         print('total time: {0}'.format(inv - start))
    #
    #
    #     thunk.inputs = inputs
    #     thunk.outputs = outputs
    #     thunk.lazy = False
    #
    #     return thunk

    def c_support_code_apply(self, node, name):
        return """
            static float* %(n)s_pivot;
            static float* %(n)s_info;
            static float** %(n)s_x_list;
            static float** %(n)s_z_list;
            static float** %(n)s_x_gpu;
            static float** %(n)s_z_gpu;

            static size_t %(n)s_pivot_size;
            static size_t %(n)s_info_size;
            static size_t %(n)s_list_len;

            static int %(n)s_prep(int batch_size, int size)
            {
                int p_s = batch_size * size;
                if(%(n)s_pivot_size != p_s || %(n)s_info_size != batch_size)
                {
                    if(%(n)s_pivot) device_free(%(n)s_pivot);
                    if(%(n)s_info) device_free(%(n)s_info);


                    %(n)s_pivot = (float *) device_malloc(p_s * sizeof(float));
                    %(n)s_info = (float *) device_malloc(batch_size * sizeof(float));

                    float *host_info = (float *) malloc(batch_size * sizeof(float));

                    for(float* temp = host_info, int i = 0; i < batch_size; i++)
                    {
                        *temp = 0;
                        ++temp;
                    }

                    cudaError_t err1;
                    err1 = cudaMemcpy(%(n)s_info, host_info, batch_size * sizeof(float), cudaMemcpyHostToDevice);
                    free(host_info);
                    if (err1 != cudaSuccess)
                    {
                        return -1;
                    }

                    %(n)s_pivot_size = p_s;
                    %(n)s_info_size = batch_size;

                }

                if(%(n)s_list_len != batch_size)
                {
                    if(%(n)s_x_list) free(%(n)s_x_list);
                    if(%(n)s_z_list) free(%(n)s_z_list);
                    if(%(n)s_x_gpu) device_free(%(n)s_x_gpu);
                    if(%(n)s_z_gpu) device_free(%(n)s_z_gpu);


                    %(n)s_x_list = (float **) malloc(batch_size * sizeof(float *));
                    %(n)s_z_list = (float **) malloc(batch_size * sizeof(float *));
                    %(n)s_x_gpu = (float **) device_malloc(batch_size * sizeof(float *));
                    %(n)s_z_gpu = (float **) device_malloc(batch_size * sizeof(float *));
                    %(n)s_list_len = batch_size;
                }

                return 1;
            }
        """ % dict(n=name)

    def c_code(self, node, name, inputs, outputs, sub):
        bx, = inputs
        bz, = outputs
        fail = sub['fail']
        # //CudaNdarray* pivot_alloc = (CudaNdarray *)CudaNdarray_New();
        # //CudaNdarray* info_alloc = NULL;
        #
        #
        # //CudaNdarray_alloc_contiguous(pivot_alloc, 2, pivot_dim);
        # //info_alloc = (CudaNdarray*) CudaNdarray_ZEROS(1, &x_dim0);
        return """
            int i, x_dim0, x_dim1, x_dim2;
            int x_stride, z_stride;
            int out_dim[3];
            int pivot_dim[2];
            int ptr_array_size = CudaNdarray_HOST_DIMS(%(bx)s)[0] * sizeof(float *);

            cublasStatus_t err;
            cudaError_t err1;

            x_dim0 = CudaNdarray_HOST_DIMS(%(bx)s)[0];
            x_dim1 = CudaNdarray_HOST_DIMS(%(bx)s)[1];
            x_dim2 = CudaNdarray_HOST_DIMS(%(bx)s)[2];

            out_dim[0] = x_dim0;
            out_dim[1] = x_dim1;
            out_dim[2] = x_dim2;

            if ( !(%(bz)s
               && %(bz)s->nd==3
               && CudaNdarray_is_c_contiguous(%(bz)s)
               && CudaNdarray_HOST_DIMS(%(bz)s)[0]==out_dim[0]
               && CudaNdarray_HOST_DIMS(%(bz)s)[1]==out_dim[1]
               && CudaNdarray_HOST_DIMS(%(bz)s)[2]==out_dim[2]))
            {
                Py_XDECREF(%(bz)s);
                %(bz)s = (CudaNdarray*)CudaNdarray_NewDims(3,out_dim);
                if (NULL == %(bz)s)
                {
                    PyErr_Format(PyExc_RuntimeError,
                            "Failed to allocate output of %%d x %%d x %%d",
                            out_dim[0], out_dim[1], out_dim[2]);
                    %(fail)s;
                }
            }

            if (x_dim0 != 0 && x_dim1 != 0 && x_dim2 != 0)
            {

                %(name)s_prep(x_dim0, x_dim1);

                x_stride = CudaNdarray_HOST_STRIDES(%(bx)s)[0];
                z_stride = CudaNdarray_HOST_STRIDES(%(bz)s)[0];

                %(name)s_x_list[0] = CudaNdarray_DEV_DATA(%(bx)s);
                %(name)s_z_list[0] = CudaNdarray_DEV_DATA(%(bz)s);

                for (i = 1; i < out_dim[0]; i++)
                {
                    %(name)s_x_list[i] = %(name)s_x_list[i - 1] + x_stride;
                    %(name)s_z_list[i] = %(name)s_z_list[i - 1] + z_stride;
                }

                err1 = cudaMemcpy(%(name)s_x_gpu, %(name)s_x_list, ptr_array_size, cudaMemcpyHostToDevice);

                if (err1 != cudaSuccess)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                "%%s", "cudaMemcpy failure");
                    %(fail)s;
                }

                err1 = cudaMemcpy(%(name)s_z_gpu, %(name)s_z_list, ptr_array_size, cudaMemcpyHostToDevice);

                if (err1 != cudaSuccess)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                "%%s", "cudaMemcpy failure");
                    %(fail)s;
                }

                printf("test2");

                cublasSgetrfBatched(handle, x_dim1, %(name)s_x_gpu, x_dim1, (int *)%(name)s_pivot, (int *)%(name)s_info, x_dim0);

                cublasSgetriBatched(handle, x_dim1, (const float **)%(name)s_x_gpu, x_dim1, (const int *)%(name)s_pivot, %(name)s_z_gpu, x_dim1, (int *)%(name)s_info, x_dim0);

            }

        """ % locals()

    def c_cleanup_code_struct(self, node, name):

        return """
            printf("test");
            if (%(n)s_pivot) device_free(%(n)s_pivot);
            if (%(n)s_info) device_free(%(n)s_info);
            if (%(n)s_x_list) free(%(n)s_x_list);
            if (%(n)s_z_list) free(%(n)s_z_list);
            if (%(n)s_x_gpu) device_free(%(n)s_x_gpu);
            if (%(n)s_z_gpu) device_free(%(n)s_z_gpu);
        """ % dict(n=name)

    def c_code_cache_version(self):
        return (0,1,18)

batched_inv = BatchedInvOp()
