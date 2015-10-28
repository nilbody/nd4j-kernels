#include <math.h>
#include <sharedmem.h>
#include <tad.h>


//an op for the kernel
template<typename T>
__device__ T op(T d1,T d2,T *extraParams);

//calculate an update of the reduce operation
template<typename T>
__device__ T update(T old,T opOutput,T *extraParams);


//post process result (for things like means etc)
template<typename T>
__device__ T postProcess(T reduction,int n,int xOffset,T *dx,int incx,T *extraParams,T *result);

template<typename T>
__device__ T merge(T old,T opOutput,T *extraParams);

template <typename T>
__device__ T doBlock(
		int n,
		T *sPartials,
		T *dx,
		int xOffset,
		int incx,
		T *dy,
		int yOffset,
		int incy,
		T *extraParams) {
	T reduce = extraParams[0];
	int tid = threadIdx.x;
	int totalThreads = gridDim.x * blockDim.x;
	int start = blockDim.x * blockIdx.x + tid;
	for (int i = start; i < n; i += totalThreads) {
		int currIdx = xOffset + i * incx;
		int currYIdx = yOffset + i * incy;
		T curr = dx[currIdx];
		T currY = dy[currYIdx];
		reduce = update(reduce,op(curr,currY,extraParams),extraParams);
	}

	return reduce;
}



template<typename T>
__device__ void aggregatePartials(T *sPartials,int tid,T *extraParams) {

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

}

template<typename T>
__global__ void doReduce(
		T *dx
		,T *extraParams
		,int n
		,int incx
		,int xOffset,
		T *dy,
		int incy,
		int yOffset,
		T *result,
		int resultOffset) {

	SharedMemory<T> val;
	T *sPartials = val.getPointer();
	int tid = threadIdx.x;
	T reduce = doBlock(n,sPartials,dx,xOffset,incx,dy,yOffset,incy,extraParams);
	sPartials[tid] = reduce;
	__syncthreads();

	aggregatePartials(sPartials,tid,extraParams);

	if (tid == 0) {
		result[resultOffset] = postProcess(sPartials[0],n,xOffset,dx,incx,extraParams,result);
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
__device__ void transform(
		int n
		,T *dx
		,int *xShapeInfo,
		T *dy,
		int *yShapeInfo
		,T *extraParams
		,T *result,
		int *resultShapeInfo
		,int *gpuInformation,
		int *dimension,
		int dimensionLength) {

	__shared__ int nIsPow2;
	nIsPow2 = (n % 2 == 0);
	__syncthreads();
	/**
	 * Gpu information for the problem
	 */
	int tid = threadIdx.x;


	//shared shape information for a given block
	__shared__ ShapeInformation *xInfo;
	__shared__ ShapeInformation *yInfo;
	__shared__ ShapeInformation *resultInfo;



	//setup the shared shape information
	if(tid == 0)  {
		//initialize the shape information only once
		xInfo = infoFromBuffer(xShapeInfo);
		yInfo = infoFromBuffer(yShapeInfo);
		resultInfo =  infoFromBuffer(resultShapeInfo);
	}

	__syncthreads();
	//init the tad off




	__shared__ int xLength;
	if(tid == 0) {
		//__device__ __host__ int* keep(int *data,int *index,int indexLength,int dataLength) {
		int *keep2 = keep(xInfo->shape,dimension,dimensionLength,xInfo->rank);
		xLength = prod(keep2,dimensionLength);
		free(keep2);
	}

	__syncthreads();


	__shared__ int resultScalar;
	resultScalar = isScalar(resultInfo);
	__syncthreads();

	//shared memory space for storing intermediate results
	SharedMemory<T> val;
	volatile T *sPartials = val.getPointer();
	SharedMemory<T> valY;
	volatile T *sYPartials = valY.getPointer();


	ShapeInformation *xInfoCopy = shapeCopy(xInfo);
	ShapeInformation *yInfoCopy = shapeCopy(yInfo);
	ShapeInformation *resultInfoCopy = shapeCopy(resultInfo);
	if(!resultScalar) {
		int offset3 = offset(blockIdx.x ,xInfoCopy->rank,xInfoCopy,dimension,dimensionLength);
		int offset4 = offset(blockIdx.x ,yInfoCopy->rank,yInfoCopy,dimension,dimensionLength);

		sPartials[tid] = dx[offset3 + tid * xInfoCopy->elementWiseStride];
		sYPartials[tid] = dy[offset4 + tid * yInfoCopy->elementWiseStride];

	}
	else {

		int blockSize = gpuInformation[0];

		// perform first level of reduction,
		// reading from global memory, writing to shared memory
		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
		if(i >= n)
			return;

		unsigned int gridSize = blockSize * 2 * gridDim.x;

		T reduction = extraParams[0];
		T curr,currY;

		// we reduce multiple elements per thread.  The number is determined by the
		// number of active thread blocks (via gridDim).  More blocks will result
		// in a larger gridSize and therefore fewer elements per thread
		while (i < n)	{
			curr = dx[i];
			currY = dy[i];
			reduction = update(reduction,op(curr,currY,extraParams),extraParams);


			// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
			if (nIsPow2 || i + blockSize < n) {
				curr = dx[i + blockSize];
				currY = dy[i + blockSize];
				reduction = update(reduction,op(curr,currY,extraParams),extraParams);

			}

			i += gridSize;
		}

		// each thread puts its local sum into shared memory
		sPartials[tid] = reduction;
		sYPartials[tid] = reduction;
		__syncthreads();


		// do reduction in shared mem
		if ((blockSize >= 512) && (tid < 256)) {
			curr = sPartials[tid + 256];
			currY = sYPartials[tid + 256];
			reduction = update(reduction,op(curr,currY,extraParams),extraParams);
			sPartials[tid] = reduction;
			sYPartials[tid] = reduction;
		}

		__syncthreads();

		if ((blockSize >= 256) &&(tid < 128)) {
			curr = sPartials[tid + 128];
			currY = sYPartials[tid + 128];
			reduction = update(reduction,op(curr,currY,extraParams),extraParams);
			sPartials[tid] = reduction;
		}

		__syncthreads();

		if ((blockSize >= 128) && (tid <  64)) {
			curr = sPartials[tid + 64];
			currY = sYPartials[tid + 64];
			reduction = update(reduction,op(curr,currY,extraParams),extraParams);
			sPartials[tid] = reduction;
			sYPartials[tid] = reduction;
		}

		__syncthreads();

#if (__CUDA_ARCH__ >= 300 )
		if ( tid < 32 ) {
			// Fetch final intermediate sum from 2nd warp
			if (blockSize >=  64) {
				curr = sPartials[tid + 32];
				currY = sYPartials[tid + 32];
				reduction = update(reduction,op(curr,currY,extraParams),extraParams);
				sPartials[tid] = reduction;
				sYPartials[tid] = reduction;

			}
			// Reduce final warp using shuffle
			for (int offset = warpSize/2; offset > 0; offset /= 2) {
				curr =  __shfl_down(reduction, offset);
				currY = curr;
				reduction = update(reduction,op(curr,currY,extraParams),extraParams);
			}
		}
#else
		// fully unroll reduction within a single warp
		if ((blockSize >=  64) && (tid < 32)) {
			curr = sPartials[tid + 32];
			currY = sYPartials[tid + 32];
			reduction = update(reduction,op(curr,currY,extraParams),extraParams);
			sPartials[tid] = reduction;
			sYPartials[tid] = reduction;
		}

		__syncthreads();

		if ((blockSize >=  32) && (tid < 16)) {
			curr = sPartials[tid + 16];
			currY = sYPartials[tid + 16];
			reduction = update(reduction,op(curr,currY,extraParams),extraParams);
			sPartials[tid] = reduction;
			sYPartials[tid] = reduction;

		}

		__syncthreads();

		if ((blockSize >=  16) && (tid <  8)) {
			curr = sPartials[tid + 8];
			currY = sPartials[tid + 8];
			reduction = update(reduction,op(curr,currY,extraParams),extraParams);
			sPartials[tid] = reduction;
			sYPartials[tid] = reduction;
		}

		__syncthreads();

		if ((blockSize >=   8) && (tid <  4)) {
			curr = sPartials[tid + 4];
			currY = sYPartials[tid + 4];
			reduction = update(reduction,op(curr,currY,extraParams),extraParams);
			sPartials[tid] = reduction;
			sYPartials[tid] = reduction;
		}

		__syncthreads();

		if ((blockSize >=   4) && (tid <  2)) {
			curr = sPartials[tid + 2];
			currY = sYPartials[tid + 2];
			reduction = update(reduction,op(curr,currY,extraParams),extraParams);
			sPartials[tid] = reduction;
			sYPartials[tid] = reduction;
		}

		__syncthreads();

		if ((blockSize >=   2) && ( tid <  1)) {
			curr = sPartials[tid + 1];
			currY = sYPartials[tid + 1];
			reduction = update(reduction,op(curr,currY,extraParams),extraParams);
			sPartials[tid] = reduction;
			sYPartials[tid] = reduction;
		}

		__syncthreads();
#endif

		// write result for this block to global mem
		if (tid == 0)
			result[blockIdx.x] = postProcess(reduction,n,xInfo->offset,dx,xInfo->elementWiseStride,extraParams,result);
	}
	__syncthreads();
	if(tid == 0) {
		if(!resultScalar) {
			T currResult = extraParams[0];
			for(int i = 0; i < xLength; i++) {
				currResult = update(currResult,op(sPartials[i],sYPartials[i],extraParams),extraParams);
			}

			result[blockIdx.x] = postProcess(currResult,n,xInfo->offset,dx,xInfo->elementWiseStride,extraParams,result);

		}

	}


	free(xInfoCopy);
	free(resultInfoCopy);
}

extern "C"
__global__ void transform_double(
		int n
		,double *dx
		,int *xShapeInfo,
		double *dy,
		int *yShapeInfo
		,double *extraParams
		,double *result,
		int *resultShapeInfo
		,int *gpuInformation,
		int *dimension,
		int dimensionLength) {
	transform<double>(
			n,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			gpuInformation,
			dimension,
			dimensionLength);

}


extern "C"
__global__ void transform_float(
		int n
		,float *dx
		,int *xShapeInfo,
		float *dy,
		int *yShapeInfo
		,float *extraParams
		,float *result,
		int *resultShapeInfo
		,int *gpuInformation,
		int *dimension,
		int dimensionLength) {
	transform<float>(
			n,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			gpuInformation,
			dimension,
			dimensionLength);

}


