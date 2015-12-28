#include <reduce_common.h>
#include <sharedmem.h>
#include <postprocess.h>
#include <tad.h>
#include <shape.h>


namespace nd4j {
namespace functions {
namespace reduce3 {


template <typename T>
/**
 *
 */
class BaseReduce3 : public Reduce3<T> {

public:

	/**
	 * Aggregate shared memory
	 * @param sPartialsRef
	 * @param tid
	 * @param extraParams
	 */
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


		__shared__ int *yShape;
		__shared__ int yRank;
		__shared__ int yElementWiseStride;
		__shared__ int yOffset;



		//shared memory space for storing intermediate results
		SharedMemory<T> val;
		volatile T *sPartials = val.getPointer();
		int numElements = gpuInformation[2] / sizeof(T);
		for (int i = tid; i < numElements; i += blockDim.x)
			sPartials[i] = extraParams[0];
		__syncthreads();


		sPartials[tid] = extraParams[0];
		sPartials[(1 + tid) * 2] = extraParams[0];
		__syncthreads();


		//starting index for tad
		__shared__ volatile int currentYBlockOffset;
		//ending index for tad
		__shared__ volatile int endingYOffset;
		//length for the tad
		__shared__ volatile int yLength;




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
		__shared__ TADPermuteInfo yTadInfo;
		__shared__ TADPermuteInfo resultTadInfo;

		int valueOffset,valueYOffset;

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

			yElementWiseStride = elementWiseStride(yShapeInfo);
			yOffset = offset(yShapeInfo);
		}


		__syncthreads();

		T curr,currY;

		if(resultScalar) {
			int blockSize = gpuInformation[0];

			unsigned int i = xOffset +   blockIdx.x   *  xElementWiseStride + tid;
			unsigned int j = yOffset +   blockIdx.x   *  yElementWiseStride + tid;
			unsigned int gridSize = blockDim.x * gridDim.x *  xElementWiseStride;
			unsigned int gridSizeY = blockDim.x * gridDim.x *  yElementWiseStride;


			// we reduce multiple elements per thread.  The number is determined by the
			// number of active thread blocks (via gridDim).  More blocks will result
			// in a larger gridSize and therefore fewer elements per thread
			while (i * xElementWiseStride < xLength && j * yElementWiseStride < yLength)	{
				curr = dx[i];
				currY = dy[j];
				reduction = update(reduction,op(curr,currY,extraParams),extraParams);
				i += gridSize;
				j += gridSizeY;
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
				yTadInfo  = tadInfo(yShapeInfo,dimension,dimensionLength);
				resultTadInfo = tadInfo(resultShapeInfo,dimension,dimensionLength);


				resultScalar = isScalar(resultShapeInfo);
				currentBlockOffset = offset(blockIdx.x, xShapeInfo,dimension,dimensionLength,xTadInfo);
				endingOffset = offset(blockIdx.x + 1 ,xShapeInfo,dimension,dimensionLength,xTadInfo);
				resultLength = prod(shape(resultShapeInfo),rank(resultShapeInfo));

				//initialize x
				xShape = shape(xShapeInfo);
				xRank = rank(xShapeInfo);
				xOffset = offset(xShapeInfo);
				xElementWiseStride = elementWiseStride(xShapeInfo);


				//initialize y
				yShape = shape(yShapeInfo);
				yRank = rank(yShapeInfo);
				yOffset = offset(yShapeInfo);
				yElementWiseStride = elementWiseStride(yShapeInfo);


				currentYBlockOffset = offset(blockIdx.x, yShapeInfo,dimension,dimensionLength,yTadInfo);
				endingYOffset = offset(blockIdx.x + 1 ,yShapeInfo,dimension,dimensionLength,yTadInfo);


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

			__syncthreads();

			//number of tads per block to process
			for(int i = 0; i < tadsForBlock; i++) {
				int tadIndex = tadForBlockIndex(gpuInformation[0],blockIdx.x,i);
				int blockOffset = offset(tadIndex, xShapeInfo,dimension,dimensionLength,xTadInfo);
				int blockYOffset = offset(tadIndex, yShapeInfo,dimension,dimensionLength,yTadInfo);

				//concurrently load all elements in to shared memory
				if(elementsPerThread > 1) {
					for(int i = 0; i < elementsPerThread; i++) {
						if(i > 0) {
							valueOffset = blockOffset  +(tid * i * xElementWiseStride);
							valueYOffset = blockYOffset + (tid * i * yElementWiseStride);
							//break at the end
							if(valueOffset >= n)
								break;
							T val = dx[valueOffset];
							T yVal = dy[valueYOffset];
							sPartials[tid] = update(sPartials[tid],op(val,yVal,extraParams),extraParams);
						}

						else {
							valueOffset = blockOffset  +(tid * i * xElementWiseStride);
							valueYOffset = blockYOffset + (tid * i * yElementWiseStride);

							//break at the end
							if(valueOffset >= n)
								break;
							T val = dx[valueOffset];
							T yVal = dy[valueYOffset];
							printf("Comparing value x %f and y %f\n",val,yVal);
							sPartials[tid] = val;
							sPartials[(1 + tid)* 2] = yVal;
						}



					}
				}
				else {
					int blockOffset = currentBlockOffset;
					int yBlockOffset = currentYBlockOffset;
					valueOffset = blockOffset  + tid * xElementWiseStride;
					valueYOffset = yBlockOffset + tid * yElementWiseStride;
					T val = dx[valueOffset];
					T val2 = dy[valueYOffset];
					sPartials[tid] = val;
					sPartials[(1 + tid) * 2] = val2;
				}

				__syncthreads();

				//do reduction in shared memory only on the first thread
				if(tid == 0) {
					curr = startValue;
					for(int j = 0; j < xLength; j++) {
						curr = update(curr,op(sPartials[j],sPartials[(1 + j) * 2],extraParams),extraParams);
					}

					if(postProcessOrNot) {
						result[tadIndex] = postProcess(curr,xLength,xOffset,dx, xElementWiseStride,extraParams,result);
					}
					else {
						result[tadIndex] = curr;
					}
				}
			}


		}



		if(resultScalar && tid == 0) {
			freePermuteInfo(xTadInfo);
			freePermuteInfo(yTadInfo);
			freePermuteInfo(resultTadInfo);
		}


	}

};

}
}
}
