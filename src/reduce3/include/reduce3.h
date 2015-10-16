#include <math.h>
#include <sharedmem.h>



//an op for the kernel
template<typename T>
__device__ T op(T d1,T d2,T *extraParams);

//calculate an update of the reduce operation
template<typename T>
__device__ T update(T old,T opOutput,T *extraParams);


//post process result (for things like means etc)
template<typename T>
__device__ T postProcess(T reduction,int n,int xOffset,T *dx,int incx,T *extraParams,T *result);

/**

Perform a reduction
@param n the number of elements
@param xOffset the starting offset
@param dx the data to perform the reduction on
@param incx the increment on which to perform the reduction
@param extraParams extra parameters used for calculations
@param result where to store the result of the reduction
 */
template<typename T>
__device__ void transform_pair(int n, int xOffset,int yOffset,T *dx,T *dy,int incx,int incy,T *extraParams,T *result,int i,int blockSize) {
	struct SharedMemory<T> val;
	T *sPartials = val.getPointer();
	const int tid = threadIdx.x;
	if(tid >= blockSize)
		return;
	int totalThreads = gridDim.x * blockDim.x;
	int start = blockDim.x * blockIdx.x + tid;

	T sum = extraParams[0];
	for (size_t i = start; i < n; i += totalThreads) {
		sum = update(sum,op(dx[i * incx],dy[i * incy],extraParams),extraParams);
	}

	sPartials[tid] = sum;
	__syncthreads();

	// start the shared memory loop on the next power of 2 less
	// than the block size.  If block size is not a power of 2,
	// accumulate the intermediate sums in the remainder range.
	int floorPow2 = blockDim.x;

	if ( floorPow2 & (floorPow2 - 1) ) {
		while ( floorPow2 & (floorPow2 - 1) ) {
			floorPow2 &= floorPow2 - 1;
		}
		if ( tid >= floorPow2 ) {
			sPartials[tid - floorPow2] = update(sPartials[tid - floorPow2],sPartials[tid - floorPow2],extraParams);
		}
		__syncthreads();
	}

	for ( int activeThreads = floorPow2 >> 1;
			activeThreads;
			activeThreads >>= 1 ) {
		if (tid < activeThreads) {
			sPartials[tid] = update(sPartials[tid],sPartials[tid + activeThreads],extraParams);
		}
		__syncthreads();
	}

	if ( tid == 0 ) {
		result[i] = postProcess(sPartials[0],n,xOffset,dx,incx,extraParams,result);
	}

}


/**

Perform a reduction
@param n the number of elements
@param xOffset the starting offset
@param dx the data to perform the reduction on
@param incx the increment on which to perform the reduction
@param extraParams extra parameters used for calculations
@param result where to store the result of the reduction
 */
template<typename T>
__device__ void transform(int n, int xOffset,T *dx,int incx,T *extraParams,T *result,int i2,int blockSize) {
	SharedMemory<T> val;
	T *sPartials = val.getPointer();
	const int tid = threadIdx.x;
	int totalThreads = gridDim.x * blockDim.x;
	int start = blockDim.x * blockIdx.x + tid;

	T sum = result[0];
	for ( size_t i = start; i < n; i += totalThreads) {
		sum = update(sum,op(dx[i * incx],sum,extraParams),extraParams);
	}

	sPartials[tid] = sum;
	__syncthreads();

	// start the shared memory loop on the next power of 2 less
	// than the block size.  If block size is not a power of 2,
	// accumulate the intermediate sums in the remainder range.
	int floorPow2 = blockDim.x;

	if (floorPow2 & (floorPow2 - 1)) {
		while (floorPow2 & (floorPow2 - 1)) {
			floorPow2 &= floorPow2 - 1;
		}
		if (tid >= floorPow2) {
			sPartials[tid - floorPow2] = update(sPartials[tid - floorPow2],op(sPartials[tid - floorPow2],sPartials[tid],extraParams),extraParams);
		}
		__syncthreads();
	}

	for (int activeThreads = floorPow2 >> 1;
			activeThreads;
			activeThreads >>= 1 ) {
		if (tid < activeThreads) {
			sPartials[tid] = update(sPartials[tid],op(sPartials[tid],sPartials[tid + activeThreads],extraParams),extraParams);
		}
		__syncthreads();
	}

	if (tid == 0) {
		result[i2] = postProcess(sPartials[0],n,xOffset,dx,incx,extraParams,result);
	}

}


template __device__ void transform_pair<double>(int n, int xOffset,int yOffset,double *dx,double *dy,int incx,int incy,double *extraParams,double *result,int i,int blockSize);
template __device__ void transform_pair<float>(int n, int xOffset,int yOffset,float *dx,float *dy,int incx,int incy,float *extraParams,float *result,int i,int blockSize);

