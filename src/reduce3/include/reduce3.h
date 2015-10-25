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
__device__ void transform_pair(
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

	//shared shape information for a given block
	__shared__ ShapeInformation *xInfo;
	__shared__ ShapeInformation *yInfo;
	__shared__ ShapeInformation *resultInfo;

	/**
	 * Gpu information for the problem
	 */
	int tid = threadIdx.x;

	//setup the shared shape information
	if(tid == 0)  {
		//initialize the shape information only once
		xInfo = infoFromBuffer(xShapeInfo);
		yInfo = infoFromBuffer(yShapeInfo);
		resultInfo =  infoFromBuffer(resultShapeInfo);
	}

	__syncthreads();



	//of note here: the tad dimensions should be the same for doing reductions across pair wise tensors
	int xLength = prod(xInfo->shape,xInfo->rank);
	int tensorsAlongDimension2 = tensorsAlongDimension(xInfo->rank,xLength,xInfo->shape,dimension,dimensionLength);



	/**
	 * Kernel function invocation
	 * information
	 */
	int sharedMemorySize = gpuInformation[2];

	//do the problem in line
	if(tensorsAlongDimension2 == 1 || isVector(xInfo->shape,xInfo->rank)) {
		int resultOffset = resultInfo->offset;
		//the overall result
		//shared memory space for storing intermediate results
		SharedMemory<T> val;
		T *sPartials = val.getPointer();
		T reduce = doBlock(n,sPartials,dx,xInfo->offset,xInfo->elementWiseStride,dy,yInfo->offset,yInfo->elementWiseStride,extraParams);
		//result for the block
		sPartials[tid] = reduce;
		__syncthreads();
		aggregatePartials(sPartials,tid,extraParams);
		if (tid == 0) {
			result[resultOffset] = postProcess(sPartials[0],n,xInfo->offset,dx,xInfo->elementWiseStride,extraParams,result);
		}
	}
	else {

		int *tadShape = removeIndex(xInfo->shape,dimension,xInfo->rank,dimensionLength);
		if(xInfo->rank - dimensionLength < 2) {
			int *newShape = ensureVectorShape(tadShape,dimension[0]);
			free(tadShape);
			tadShape = newShape;
		}


		int *keepShape = keep(xInfo->shape,dimension,dimensionLength,xInfo->rank);

		int elementsPerVector = prod(tadShape,xInfo->rank - dimensionLength);
		int numElementsToUse = isVector(tadShape,xInfo->rank - dimensionLength) ? n : prod(keepShape,dimensionLength);
		free(keepShape);
		//launch a kernel per tensor along dimension
		for(int i = 0; i < tensorsAlongDimension2; i++) {
			ShapeInformation *xInfoCopy = shapeCopy(xInfo);
			ShapeInformation *yInfoCopy = shapeCopy(yInfo);
			ShapeInformation *resultInfoCopy = shapeCopy(resultInfo);

			int startOffset = offset(i,xInfoCopy->rank,xInfoCopy,dimension,dimensionLength);
			int yOffset = offset(i,yInfoCopy->rank,yInfoCopy,dimension,dimensionLength);
			int resultOffset = offset(i,resultInfo->rank,resultInfo,dimension,dimensionLength);

			doReduce<<<1,1,sharedMemorySize>>>(
					dx,
					extraParams,
					numElementsToUse,
					xInfoCopy->elementWiseStride,
					startOffset,
					dy,
					yInfoCopy->elementWiseStride,
					yOffset,
					result,
					resultOffset
			);

			//only free it if it hasn't already been freed
			if(xInfo->rank - dimensionLength < 2)
				free(tadShape);
			free(xInfoCopy);
			free(resultInfoCopy);
		}

		cudaDeviceSynchronize();


	}
}

extern "C"
__global__ void transform_pair_double(
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
	transform_pair<double>(
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
__global__ void transform_pair_float(
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
	transform_pair<float>(
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


