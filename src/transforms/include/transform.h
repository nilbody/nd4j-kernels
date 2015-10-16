#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//an op for the kernel

template<typename T>
__device__ T op(T d1,T *params);

template<typename T>
__device__ void transform(int n,int idx,T *dy,int incy,T *params,T *result,int blockSize) {
	int totalThreads = gridDim.x * blockDim.x;
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tid;
	/* equal, positive, non-unit increments. */
	for (; i < n; i += totalThreads) {
		result[i * incy] = op(dy[i * incy],params);
	}


}

template <> void transform<double>(int n,int idx,double *dy,int incy,double *params,double *result,int blockSize);
template <> void transform<float>(int n,int idx,float *dy,int incy,float *params,float *result,int blockSize);


template <> double op<double>(double d1,double *params);
template <> float op<float>(float d1,float *params);

