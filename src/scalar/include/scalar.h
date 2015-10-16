#include <stdio.h>
#include <stdlib.h>
//scalar and current element
template<typename T>
__device__ T op(T d1,T d2,T *params);

template<typename T>
__device__ void transform(int n, int idx,T dx,T *dy,int incy,T *params,T *result,int blockSize) {
	int totalThreads = gridDim.x * blockDim.x;
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tid;

	for (; i < n; i += totalThreads) {
		result[i * incy] = op(dx,dy[i * incy],params);
	}

}




template __device__ void transform<double>(int n, int idx,double dx,double *dy,int incy,double *params,double *result,int blockSize);

template __device__ void transform<float>(int n, int idx,float dx,float *dy,int incy,float *params,float *result,int blockSize);
