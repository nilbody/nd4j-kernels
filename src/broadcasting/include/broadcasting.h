/*
 * broadcasting.h
 *
 *  Created on: Nov 11, 2015
 *      Author: agibsonccc
 */

#ifndef BROADCASTING_H_
#define BROADCASTING_H_

#include <math.h>

#include <sharedmem.h>
#include <tad.h>
#include <stdio.h>

template <typename T>
__device__ T op(T d1,T d2);
template <typename T>
__device__ T op(T d1);




template <typename T>
__device__ void transform(
		T *x
		,int *xShapeInfo
		,T *y
		,int *yShapeInfo
		,T *result
		,int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int *gpuInformation) {


	/**
	 * Gpu information for the problem
	 */
	int tid = threadIdx.x;

	__shared__ int *xShape;
	__shared__ int xRank;
	__shared__ int xElementWiseStride;


	__shared__ int *yShape;
	__shared__ int yRank;
	__shared__ int yElementWiseStride;
	__shared__ int yOffset;


	__shared__ int *resultShape;
	__shared__ int resultRank;
	__shared__ int resultElementWiseStride;
	__shared__ int resultOffset;

	//shared memory space for storing intermediate results
	SharedMemory<T> val;
	volatile T *sPartials = val.getPointer();
	sPartials[tid] = 0.0;
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






	if(tid == 0) {
		xTadInfo  = tadInfo(xShapeInfo,dimension,dimensionLength);
		yTadInfo  = tadInfo(yShapeInfo,dimension,dimensionLength);
		resultTadInfo = tadInfo(resultShapeInfo,dimension,dimensionLength);


		currentBlockOffset = offset(blockIdx.x, xShapeInfo,dimension,dimensionLength,xTadInfo);
		endingOffset = offset(blockIdx.x + 1 ,xShapeInfo,dimension,dimensionLength,xTadInfo);
		resultLength = prod(shape(resultShapeInfo),rank(resultShapeInfo));

		//initialize x
		xShape = shape(xShapeInfo);
		xRank = rank(xShapeInfo);
		xElementWiseStride = elementWiseStride(xShapeInfo);


		//initialize y
		yShape = shape(yShapeInfo);
		yRank = rank(yShapeInfo);
		yOffset = offset(yShapeInfo);
		yElementWiseStride = elementWiseStride(yShapeInfo);


		//initialize result
		resultShape = shape(resultShapeInfo);
		resultRank = rank(resultShapeInfo);
		resultOffset = offset(resultShapeInfo);
		resultElementWiseStride = elementWiseStride(resultShapeInfo);


		currentYBlockOffset = offset(blockIdx.x, yShapeInfo,dimension,dimensionLength,yTadInfo);
		endingYOffset = offset(blockIdx.x + 1 ,yShapeInfo,dimension,dimensionLength,yTadInfo);




		double tads = tensorsAlongDimension(xRank,prod(xShape,xRank),xShape,dimension,dimensionLength);
		if(gpuInformation[0] >= MAX_NUM_THREADS && tads > gpuInformation[0])
			tadsForBlock = tadsPerBlock(gpuInformation[0],tads);
		else
			tadsForBlock = 1;
		if(tadsForBlock < 1)
			tadsForBlock = 1;


		__syncthreads();



		//length of the buffer to broadcast
		int n = length(yShapeInfo);
		//printf("Running tads for block %d\n",tadsForBlock);
		//iterate over each tad that the block will process
		for(int tad = 0; tad < tadsForBlock; tad++) {
			//printf("Running tad %d on block %d\n",tad,blockIdx.x);
			//starting offset for the tad given the block for x
			int xOffset = offset(tad + blockIdx.x, xShapeInfo,dimension,dimensionLength,xTadInfo);
			//starting offset for the tad given the block for the result
			int resultOffset = offset(tad + blockIdx.x, resultShapeInfo,dimension,dimensionLength,resultTadInfo);
			int yOffset = offset(yShapeInfo);
			//printf("Offset x is %d and result offset is %d for block %d\n",xOffset,resultOffset,blockIdx.x);
			int resultCount = 0;
			for(;resultCount < n; resultCount++) {
				result[resultOffset + resultCount * resultElementWiseStride] =
						op(x[xOffset + resultCount * xElementWiseStride],y[yOffset + resultCount * yElementWiseStride]);
			}

		}
	}
}

extern "C"
__global__ void transform_double(double *x,int *xShapeInfo,double *y,int *yShapeInfo,double *result,int *resultShapeInfo,int *dimension,
		int dimensionLength,
		int *gpuInformation) {
	transform<double>(x,xShapeInfo,y,yShapeInfo,result,resultShapeInfo,dimension,dimensionLength,gpuInformation);
}


extern "C"
__global__ void transform_float(float *x,int *xShapeInfo,float *y,int *yShapeInfo,float *result,int *resultShapeInfo,int *dimension,
		int dimensionLength,
		int *gpuInformation) {
	transform<float>(x,xShapeInfo,y,yShapeInfo,result,resultShapeInfo,dimension,dimensionLength,gpuInformation);
}



#endif /* BROADCASTING_H_ */
