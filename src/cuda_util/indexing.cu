/*
 * indexing_impl.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef INDEXING_IMPL_H_
#define INDEXING_IMPL_H_
#include "indexing.h"
namespace nd4j {
namespace indexing {
/**
 * Returns the number of elements per thread
 */
__device__ int numElementsPerThread(int N) {
	return (gridDim.x * blockDim.x);
}


/**
 * Returns the block starting index
 */
__device__ int blockStartingIndex(int N) {
	return numElementsPerThread(N) * blockIdx.x * blockDim.x;
}


/**
 * Returns the thread starting index
 */
__device__ int threadStartingIndex(int N,int stride,int offset) {
	return  blockStartingIndex(N)
			+ (threadIdx.x / stride) * numElementsPerThread(N) * stride
			+ ((threadIdx.x + offset) % stride);
}



/**
 * Returns the thread ending index
 */
__device__ int threadEndingIndex(int N,int stride,int offset) {
	int ret =  threadStartingIndex(N,stride,offset) + numElementsPerThread(N) * stride;
	if(ret > N)
		ret = N;
	return ret;
}


/**
 * Returns indexing information
 * for the current kernel invocation
 */
__device__ CurrentIndexing* currentIndex(int N,int offset,int stride) {
	int numElementsPerThread = N / (gridDim.x * blockDim.x);
	if(numElementsPerThread < 1)
		numElementsPerThread = 1;
	int blockStartingIndex = numElementsPerThread * blockIdx.x * blockDim.x;
	int startingThreadIndex = blockStartingIndex
			+ (threadIdx.x / stride) * numElementsPerThread * stride
			+ ((threadIdx.x + offset) % stride);
	int endingThreadIndex = startingThreadIndex + numElementsPerThread * stride;
	if(endingThreadIndex > N)
		endingThreadIndex = N;
	CurrentIndexing *ret = (CurrentIndexing *) malloc(sizeof(CurrentIndexing));
	ret->blockStartingIndex = blockStartingIndex;
	ret->endingThreadIndex = endingThreadIndex;
	ret->numElementsPerThread = numElementsPerThread;
	ret->startingThreadIndex = startingThreadIndex;
	return ret;
}

}
}

#endif /* INDEXING_IMPL_H_ */
