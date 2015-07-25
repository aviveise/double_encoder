git pull git://github.com/aviveise/double_encoder.git

dataset="$1"
strategy="$2"
top="$3"

export PATH=/usr/local/cuda-6.5/bin/:/usr/java/jdk1.7.0_51/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/lib/atlas-base/:/usr/lib/atlas-base/atlas
export LD_LIBRARY_PATH=/usr/local/cuda-6.5/lib64:/usr/local/cuda-6.5/lib64/lib:/opt/opencv/lib:/usr/local/lib:/opt/intel/mkl/lib/intel64:/opt/tbb/lib/intel64:/usr/lib/atlas-base/:/usr/lib/atlas-base/atlas:
export CUDA_ROOT=/usr/local/cuda-6.5/bin

if [ "$strategy" = "iterative" ]; then
	THEANO_FLAGS=optimization_including=cudnn,mode=FAST_RUN,device=gpu,floatX=float32,"blas.ldflags=-lblas -lgfortran",optimizer=fast_compile python2.7 -u ./lib/double_encoder_iterative.py DataSet/$dataset.ini test_$dataset.ini $top
elif [ "$strategy" = "dropout" ]; then
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python2.7 ./lib/double_encoder_iterative_dropout.py DataSet/$dataset.ini test_$dataset.ini $top
elif [ "$strategy" = "nonsequential" ]; then
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,"blas.ldflags=-lblas -lgfortran",optimizer=fast_compile nohup python2.7 ./lib/double_encoder_iterative_nonsequential.py DataSet/$dataset.ini test_$dataset.ini $top 
else
	echo "unknown strategy"
fi
