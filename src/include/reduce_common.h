/*
 * reduce_common.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef REDUCE_COMMON_H_
#define REDUCE_COMMON_H_

/** Given an linear index, element wise stride
 * and the length of each tad
 * map a linear index to a tad
 * @param i the index to map
 * @param the element wise stride for the tads
 * @param numElementsPerTad the number of elements
 * per tad
 */
__device__ __host__ int tadIndex(int i,int elementWiseStride,int numElementsPerTad);

/**
 * Map a tad to a
 * reduction index.
 * @param tadIndexForOriginal the original tad index for the
 * split up problem (eg: split is dimension 3 mapping to a 2,3 problem)
 * @param tadsForReduced the number of tads for the shrunk down problem (eg: 2,3)
 * @param tadsForOriginal the number of tads for the smaller problem (eg: 3)
 */
__device__ __host__ int reductionIndexForTad(int tadIndexForOriginal,int tadsForReduced,int tadsForOriginal);

/**
 * Computes the number of tads
 * per reduce index for the
 * reduction tad.
 */
__device__ __host__ int tadsPerReduceIndex(int tadsForReduce,int tadsForOriginal);

/**
 * Maps a linear index to a reduction index
 * @param i the linear index to map
 * @param elementWiseStride the element wise stride
 * for the multiple problem
 * @param tadNum the number of tads for the shrunken problem
 * @param originalTadNum the tad number for the reduced version of the problem
 */
__device__ __host__ int reductionIndexForLinear(
		int i
		,int elementWiseStride
		,int numElementsPerTad
		,int tadNum
		,int originalTadNum);

#endif /* REDUCE_COMMON_H_ */
