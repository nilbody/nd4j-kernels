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


	/**
	 * Gpu information for the problem
	 */
	int tid = threadIdx.x;




	__shared__ volatile int elementsPerThread;


	//only compute the tad indexes once
	__shared__ TADPermuteInfo xTadInfo;
	__shared__ TADPermuteInfo yTadInfo;
	__shared__ TADPermuteInfo resultTadInfo;


	__shared__ int tads;




	if(tid == 0) {
		xTadInfo  = tadInfo(xShapeInfo,dimension,dimensionLength);
		yTadInfo  = tadInfo(yShapeInfo,dimension,dimensionLength);
		resultTadInfo = tadInfo(resultShapeInfo,dimension,dimensionLength);
		tads = tensorsAlongDimension(rank(xShapeInfo),length(xShapeInfo),shape(xShapeInfo),dimension,dimensionLength);
		elementsPerThread = 1;
		if(blockDim.x >= MAX_THREADS_PER_BLOCK) {
			elementsPerThread = length(xShapeInfo) / blockDim.x;
		}
	}


	__syncthreads();



	if(tid >= length(yShapeInfo))
		return;

	if(elementsPerThread > 1) {
		for (int i = 0; i < tads; i++) {
			for(int j = 0; j < elementsPerThread; j += blockDim.x) {
				int xOffset2  = offset(i, xShapeInfo,dimension,dimensionLength,xTadInfo);
				int xIdx = xOffset2 +  (tid + j) * elementWiseStride(xShapeInfo);
				if(length(xShapeInfo) <= xIdx)
					break;
				int resultIdx = xIdx;
				result[resultIdx] = op(x[xIdx],y[tid * elementWiseStride(yShapeInfo)]);

			}

		}
	}
	else {
		for (int i = 0; i < tads; i++) {
			int xOffset2  = offset(i, xShapeInfo,dimension,dimensionLength,xTadInfo);
			int xIdx = xOffset2 +  tid * elementWiseStride(xShapeInfo);
			if(length(xShapeInfo) <= xIdx)
				break;
			int resultIdx = xIdx;
			result[resultIdx] = op(x[xIdx],y[tid * elementWiseStride(yShapeInfo)]);

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
