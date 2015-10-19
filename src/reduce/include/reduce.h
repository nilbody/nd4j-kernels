#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sharedmem.h>



//an op for the kernel
template<typename T>
__device__ T op(T d1,T *extraParams);

//calculate an update of the reduce operation
template<typename T>
__device__ T update(T old,T opOutput,T *extraParams);
//invoked when combining two kernels
template<typename T>
__device__ T merge(T f1, T f2,T *extraParams);

//post process result (for things like means etc)
template<typename T>
__device__ T postProcess(T reduction,int n,int xOffset,T *dx,int incx,T *extraParams,T *result);


template<typename T>
__device__ T op(T d1,T d2,T *extraParams);


template <>  double merge<double>(double  opOutput,double other,double *extraParams);
template <> double update<double>(double old,double opOutput,double *extraParams);
template <> double op<double>(double d1,double *extraParams);
template <> double postProcess<double>(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result);


template <> float merge<float>(float old,float opOutput,float *extraParams);
template <>float update<float>(float old,float opOutput,float *extraParams);
template <> float op<float>(float d1,float *extraParams);
template <> float postProcess<float>(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result);





template<typename T>
struct ReduceFunction {
	typedef void (*reduceFunction)(
			T *dx
			,T *extraParams
			,int n
			,int incx,
			int xOffset
			,T *result
			,T *resultOffset);
};


template<typename T>
__global__ void doReduce(
		T *dx
		,T *extraParams
		,int n
		,int incx
		,int xOffset,
		T *result,
		int resultOffset) {
	T sum = extraParams[0];
	SharedMemory<T> val;
	T *sPartials = val.getPointer();
	int tid = threadIdx.x;
	int totalThreads = gridDim.x * blockDim.x;
	int start = blockDim.x * blockIdx.x + tid;

	for ( int i = start; i < n; i += totalThreads) {
		int currIdx = xOffset + i * incx;
		T curr = dx[currIdx];
		sum = update(sum,op(curr,extraParams),extraParams);


	}

	sPartials[tid] = sum;
	__syncthreads();

	// start the shared memory loop on the next power of 2 less
	// than the block size.  If block size is not a power of 2,
	// accumulate the intermediate sums in the remainder range.
	int floorPow2 = blockDim.x;

	if (floorPow2 & (floorPow2 - 1)) {
		while ( floorPow2 & (floorPow2 - 1) ) {
			floorPow2 &= floorPow2 - 1;
		}
		if (tid >= floorPow2) {
			sPartials[tid - floorPow2] = merge(sPartials[tid - floorPow2],sPartials[tid],extraParams);
		}
		__syncthreads();
	}

	for (int activeThreads = floorPow2 >> 1;activeThreads;	activeThreads >>= 1) {
		if (tid < activeThreads) {
			sPartials[tid] = merge(sPartials[tid],sPartials[tid + activeThreads],extraParams);
		}
		__syncthreads();
	}

	if (tid == 0) {
		result[resultOffset] = postProcess(sPartials[0],n,xOffset,dx,incx,extraParams,result);
	}
}



template<typename T>
__device__ ReduceFunction<T>  getReduceFunction();

/**
 * @param n n is the number of
 *        elements to loop through
 * @param dx the data to operate on
 * @param xVectorInfo the meta data for the vector:
 *                              0 is the offset
 *                              1 is the increment/stride
 *                              2 is the real length of the buffer (n and dx.length won't always be the same)
 *                              3 is the element wise stride for the buffer
 * @param gpuInformation
 *                              0 is the block size
 *                              1 is the grid size
 *                              2 is the shared memory size
 * @param problemDefinition
 *                          0 is the number of elements per vector
 *                          1 is the number of vectors
 */
template<typename T>
__device__ void transform(
		int n
		,T *dx
		,int *xVectorInfo
		,T *extraParams
		,T *result,
		int *resultVectorInfo
		,int *gpuInformation,
		int *problemDefinition) {


	/**
	 * Gpu information for the problem
	 */
	int tid = threadIdx.x;
	if(tid >= n)
		return;
	int totalThreads = gridDim.x * blockDim.x;
	int start = blockDim.x * blockIdx.x + tid;
	if(start >= n)
		return;


	/**
	 * Description of the problem
	 */
	int elementsPerVector = problemDefinition[0];
	int numberOfVectors = problemDefinition[1];

	/**
	 * X vector information
	 */
	int xOffset = xVectorInfo[0];
	int incx = xVectorInfo[1];
	int xLength = xVectorInfo[2];
	int xElementWiseStride = xVectorInfo[3];



	/**
	 * Result vector information
	 */
	int resultOffset = resultVectorInfo[0];
	int incResult = resultVectorInfo[1];
	int  resultLength = resultVectorInfo[2];
	int resultElementWiseStride = resultVectorInfo[3];




	/**
	 * Kernel function invocation
	 * information
	 */
	int blockSize = gpuInformation[0];
	int gridSize = gpuInformation[1];
	int sharedMemorySize = gpuInformation[2];
	int streamToExecute = gpuInformation[3];

	//do the problem in line
	if(numberOfVectors == 1) {
		int resultOffset = resultVectorInfo[0];
		//the overall result
		T sum = extraParams[0];
		//shared memory space for storing intermediate results
		SharedMemory<T> val;
		T *sPartials = val.getPointer();
		//actual reduction loop
		for (int i = start; i < n; i += totalThreads) {
			T curr = dx[i * incx];
			sum = update(sum,op(curr,extraParams),extraParams);
		}

		//result for the block
		sPartials[tid] = sum;
		__syncthreads();

		// start the shared memory loop on the next power of 2 less
		// than the block size.  If block size is not a power of 2,
		// accumulate the intermediate sums in the remainder range.
		int floorPow2 = blockDim.x;

		if (floorPow2 & (floorPow2 - 1)) {
			while ( floorPow2 & (floorPow2 - 1) ) {
				floorPow2 &= floorPow2 - 1;
			}
			if (tid >= floorPow2) {
				sPartials[tid - floorPow2] = merge(sPartials[tid - floorPow2],sPartials[tid],extraParams);
			}
			__syncthreads();
		}

		for (int activeThreads = floorPow2 >> 1;activeThreads;	activeThreads >>= 1) {
			if (tid < activeThreads) {
				sPartials[tid] = merge(sPartials[tid],sPartials[tid + activeThreads],extraParams);
			}
			__syncthreads();
		}

		if (tid == 0) {
			result[resultOffset] = postProcess(sPartials[0],n,xOffset,dx,incx,extraParams,result);
		}
	}
	else {

		int vectorStride = resultElementWiseStride * elementsPerVector;

		for(int i = 0; i < numberOfVectors; i++) {
			doReduce<<<blockSize,gridSize,sharedMemorySize>>>(
					dx
					,extraParams
					,elementsPerVector
					,xElementWiseStride
					,xOffset + i * vectorStride
					,result
					,resultOffset + i
			);

		}



		cudaDeviceSynchronize();


	}


}


