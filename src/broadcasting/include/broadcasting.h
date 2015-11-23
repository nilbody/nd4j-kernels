/*
 * broadcasting.h
 *
 *  Created on: Nov 11, 2015
 *      Author: agibsonccc
 */

#ifndef BROADCASTING_H_
#define BROADCASTING_H_

#include <math.h>
#include <stdio.h>

#include <sharedmem.h>
#include <tad.h>
#include <indexing.h>



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

	int length2 = shapeInfoLength(rank(xShapeInfo));


	/**
	 * Gpu information for the problem
	 */
	volatile int tid = threadIdx.x;


	volatile __shared__ int *xShape;
	volatile __shared__ int xRank;
	volatile __shared__ int xElementWiseStride;
	volatile __shared__ int xOffset;


	__shared__ int yElementWiseStride;
	__shared__ int yOffset;



	//length for the tad
	volatile __shared__  int yLength;

	//length for the tad
	volatile __shared__  int xLength;



	volatile __shared__  int resultLength;

	volatile __shared__  int tadsForBlock;

	volatile __shared__  int elementsPerThread;


	//only compute the tad indexes once
	__shared__ TADPermuteInfo xTadInfo;



	//number of times a loop through the broadcast will be made wrt the length of y and x
	volatile __shared__ int tads;




	if(tid == 0) {
		xTadInfo  = tadInfo(xShapeInfo,dimension,dimensionLength);
		resultLength = length(resultShapeInfo);
		//initialize x
		xShape = shape(xShapeInfo);
		xOffset = offset(xShapeInfo);
		xRank = rank(xShapeInfo);
		xLength = length(xShapeInfo);
		xElementWiseStride = elementWiseStride(xShapeInfo);
		//initialize y
		yLength = length(yShapeInfo);

		yOffset = offset(yShapeInfo);
		yElementWiseStride = elementWiseStride(yShapeInfo);
	}


	__syncthreads();


	int totalThreads = gridDim.x * blockDim.x;
	int i = blockIdx.x * blockDim.x + tid;
	for (; i < xLength; i += totalThreads) {
		int yOffset2 = yOffset + ((i / xElementWiseStride)% yLength) * yElementWiseStride;
        T val = x[i];
        T yVal = y[yOffset2];
        result[i] = op(val,yVal);
	}

	if(tid == 0) {
		freePermuteInfo(xTadInfo);
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
