#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sharedmem.h>
#include <tad.h>
#include <indexing.h>



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
__device__ void aggregatePartials(T **sPartialsRef,int tid,T *extraParams) {
	// start the shared memory loop on the next power of 2 less
	// than the block size.  If block size is not a power of 2,
	// accumulate the intermediate sums in the remainder range.
	T *sPartials = *sPartialsRef;
	int floorPow2 = blockDim.x;

	if (floorPow2 & (floorPow2 - 1)) {
		while ( floorPow2 & (floorPow2 - 1) ) {
			floorPow2 &= floorPow2 - 1;
		}
		if (tid >= floorPow2) {
			sPartials[tid - floorPow2] = update(sPartials[tid - floorPow2],sPartials[tid],extraParams);
		}
		__syncthreads();
	}

	for (int activeThreads = floorPow2 >> 1;activeThreads;	activeThreads >>= 1) {
		if (tid < activeThreads) {
			sPartials[tid] = update(sPartials[tid],sPartials[tid + activeThreads],extraParams);
		}
		__syncthreads();
	}

	printf("Aggregate %f for tid %d\n",sPartials[tid],tid);
}




template<typename T>
__device__ T doBlock(int n,T *sPartials,T *dx,int xOffset,int incx,T *extraParams) {
	T sum = extraParams[0];
	for (int i = 0; i < n; i++) {
		int currIdx = xOffset + i * incx;
		T curr = dx[currIdx];
		sum = update(sum,op(curr,extraParams),extraParams);
	}

	return sum;
}







template <typename T>
__device__ void initializeShared(T *extraParams,T** sPartials,int sMemSize) {
	int sPartialsLength = sMemSize / sizeof(T);
	T *sPartialsDeref = (T *) *sPartials;
	for(int i = 0; i < sPartialsLength; i++) {
		sPartialsDeref[i] = extraParams[0];
	}
}



extern "C"
__global__ void printShapeBuffer(int n,int *buff) {
	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tid;
	if(i < n) {
		printf("Buff item %d is %d\n",i,buff[i]);
	}
}


/**
 * @param n n is the number of
 *        elements to loop through
 * @param dx the data to operate on
 * @param xVectorInfo the meta data for the vector:
 *                              0 is the offset
 *                              1 is the increment/stride
 *                              2 is the real length of the buffer (n and dx.length won't always be the same)
 *                              3 is the element wise stride for the buffer
 *                              4 is the number of elements it takes to get to the next row/column/tensor
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
		,int *xShapeInfo
		,T *extraParams
		,T *result,
		int *resultShapeInfo
		,int *gpuInformation,
		int *dimension,
		int dimensionLength,int postProcessOrNot) {


	int nIsPow2 = (n % 2 == 0);
	/**
	 * Gpu information for the problem
	 */
	int tid = threadIdx.x;


	__shared__ volatile int resultScalar;


	__shared__ int *xShape;
	__shared__ int xRank;
	__shared__ int xElementWiseStride;
	__shared__ int xOffset;


	//shared memory space for storing intermediate results
	SharedMemory<T> val;
	volatile T *sPartials = val.getPointer();
	int numElements = gpuInformation[2] / sizeof(T);
	for (int i = tid; i < numElements; i += blockDim.x)
		sPartials[i] = extraParams[0];
	__syncthreads();



	//starting index for tad
	__shared__ volatile int currentBlockOffset;
	//ending index for tad
	__shared__ volatile int endingOffset;
	//length for the tad
	__shared__ volatile int xLength;

	__shared__ volatile int resultLength;

	__shared__ volatile int tadsForBlock;

	__shared__ volatile int elementsPerThread;


	//only compute the tad indexes once
	__shared__ TADPermuteInfo xTadInfo;
	__shared__ TADPermuteInfo resultTadInfo;
	int valueOffset;

	__shared__ T startValue;




	T reduction = extraParams[0];
	if(tid == 0) {
		if(dimensionLength == 1) {
			if(dimension[0] == MAX_DIMENSION)
				resultScalar = 1;
			else
				resultScalar = 0;
		}
		else
			resultScalar = 0;
		resultLength = prod(shape(resultShapeInfo),rank(resultShapeInfo));
		xOffset = offset(xShapeInfo);
		xElementWiseStride = elementWiseStride(xShapeInfo);


	}
	__syncthreads();



	T curr;

	if(resultScalar) {
		int blockSize = gpuInformation[0];

		unsigned int i = xOffset +   blockIdx.x   *  xElementWiseStride + tid;
		unsigned int gridSize = blockDim.x * gridDim.x *  xElementWiseStride;

		// we reduce multiple elements per thread.  The number is determined by the
		// number of active thread blocks (via gridDim).  More blocks will result
		// in a larger gridSize and therefore fewer elements per thread
		while (i * xElementWiseStride < n)	{
			curr = dx[i * xElementWiseStride];
			reduction = update(reduction,op(curr,extraParams),extraParams);
			i += gridSize;
		}


		// each thread puts its local sum into shared memory
		sPartials[tid] = reduction;
		__syncthreads();

		T ** sPartialsRef = (T **) &sPartials;
		aggregatePartials(sPartialsRef,tid,extraParams);



		// write result for this block to global mem
		if (tid == 0) {
			if(postProcessOrNot)
				result[blockIdx.x] = postProcess(sPartials[0],xLength,xOffset,dx, xElementWiseStride,extraParams,result);
			else {
				result[blockIdx.x] = sPartials[0];
			}
		}
	}

	else if(!resultScalar) {
		if(tid == 0) {
			xTadInfo  = tadInfo(xShapeInfo,dimension,dimensionLength);
			resultTadInfo = tadInfo(resultShapeInfo,dimension,dimensionLength);
			resultScalar = isScalar(resultShapeInfo);
			currentBlockOffset = offset(blockIdx.x, xShapeInfo,dimension,dimensionLength,xTadInfo);
			endingOffset = offset(blockIdx.x + 1 ,xShapeInfo,dimension,dimensionLength,xTadInfo);
			resultLength = prod(shape(resultShapeInfo),rank(resultShapeInfo));
			xShape = shape(xShapeInfo);
			xRank = rank(xShapeInfo);
			xOffset = offset(xShapeInfo);
			xElementWiseStride = elementWiseStride(xShapeInfo);

			//reduction on whole buffer
			if(resultScalar)
				xLength = n;

			else
				xLength = prod(xTadInfo.tensorShape,xTadInfo.tensorShapeLength);

			valueOffset = tadOffset(xShapeInfo,currentBlockOffset);
			double tads = tensorsAlongDimension(xRank,prod(xShape,xRank),xShape,dimension,dimensionLength);
			if(gpuInformation[0] >= MAX_NUM_THREADS && tads > gpuInformation[0])
				tadsForBlock = tadsPerBlock(gpuInformation[0],tads);
			else
				tadsForBlock = 1;
			if(tadsForBlock < 1)
				tadsForBlock = 1;
			//set a constant start value
			startValue = reduction;
			//when the number of elements per tad is greater than grid size, we need to compute partial
			//reductions when initializing
			if(xLength > gpuInformation[1])
				elementsPerThread = xLength / gpuInformation[1];
			else
				elementsPerThread = 1;
		}



		//number of tads per block to process
		for(int i = 0; i < tadsForBlock; i++) {
			int tadIndex = tadForBlockIndex(gpuInformation[0],blockIdx.x,i);
			int blockOffset = offset(tadIndex, xShapeInfo,dimension,dimensionLength,xTadInfo);
			//concurrently load all elements in to shared memory
			if(elementsPerThread > 1) {
				for(int i = 0; i < elementsPerThread; i++) {
					if(i > 0) {
						valueOffset = blockOffset  +(tid * i * xElementWiseStride);
						//break at the end
						if(valueOffset >= n)
							break;
						T val = dx[valueOffset];
						sPartials[tid] = update(sPartials[tid],op(val,extraParams),extraParams);
					}

					else {
						valueOffset = blockOffset  + (tid * i * xElementWiseStride);
						//break at the end
						if(valueOffset >= n)
							break;
						T val = dx[valueOffset];
						sPartials[tid] = val;
					}



				}
			}
			else {
				int blockOffset = currentBlockOffset;
				valueOffset = blockOffset  + tid * xElementWiseStride;
				T val = dx[valueOffset];
				sPartials[tid] = val;
			}

			__syncthreads();

			//do reduction in shared memory only on the first thread
			if(tid == 0) {
				curr = startValue;
				for(int j = 0; j < xLength; j++) {
					curr = update(curr,op(sPartials[j],extraParams),extraParams);
				}
				if(postProcessOrNot)
					result[tadIndex] = postProcess(curr,xLength,xOffset,dx, xElementWiseStride,extraParams,result);
				else {
					result[tadIndex] = curr;

				}
			}


			if(blockOffset >= n)
				break;
		}


	}



	if(!resultScalar && tid == 0) {
		freePermuteInfo(xTadInfo);
		freePermuteInfo(resultTadInfo);
	}

}


//kernels defined to allow lookup by name: notice extern c
extern "C"
__global__ void transform_double(
		int n
		,double *dx
		,int *xShapeInfo
		,double *extraParams
		,double *result,
		int *resultShapeInfo
		,int *gpuInformation,
		int *dimension,
		int dimensionLength,int postProcessOrNot) {
	transform<double>(
			n,
			dx,
			xShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			gpuInformation,
			dimension,
			dimensionLength,postProcessOrNot);
}


extern "C"
__global__ void transform_float(
		int n
		,float *dx
		,int *xShapeInfo
		,float *extraParams
		,float *result,
		int *resultShapeInfo
		,int *gpuInformation,
		int *dimension,
		int dimensionLength,int postProcessOrNot) {
	transform<float>(n,
			dx,
			xShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			gpuInformation,
			dimension,dimensionLength,postProcessOrNot);
}
