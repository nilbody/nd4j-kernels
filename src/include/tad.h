/*
 * tad.h
 *
 *  Created on: Oct 21, 2015
 *      Author: agibsonccc
 */

#ifndef TAD_H_
#define TAD_H_


namespace nd4j {

namespace tad {

#include <shape.h>

/**
 * Return a copy of this array with the
 * given index omitted
 *
 * @param data  the data to copy
 * @param indexes the index of the item to remove
 * @param dataLength the length of the data array
 * @param indexesLength the length of the data array
 * @return the new array with the omitted
 *
 * item
 */
__device__ __host__ void  removeIndex(int *data,int *indexes,int dataLength,int indexesLength,int **out);


/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
__device__ int tadOffset(nd4j::shape::ShapeInformation *xInfo,int offset);

/**
 * Returns a shape
 * forces the given length to be 2.
 * @param shape the shape to modify
 * @param dimension the dimension (row or column)
 * for the shape to be returned as
 * @return the new shape
 */
__device__ __host__ int*  ensureVectorShape(int *shape,int dimension);


/**
 * Generate an int buffer
 * up to the given length
 * at the specified increment
 *
 */
__device__ __host__ int* range(int from,int to,int increment);

/**
 * Range between from and two with an
 * increment of 1
 */
__device__ __host__ int* range(int from,int to);



/**
 * Keep the given indexes
 * in the data
 */
__device__ __host__ int* keep(volatile int *data,int *index,int indexLength,int dataLength);


/**
 * Generate reverse copy of the data
 * @param data
 * @param length
 * @return
 */
__device__ __host__ int* reverseCopy(int *data,int length);


/**
 *
 * @param arr1
 * @param arr1Length
 * @param arr2
 * @param arr2Length
 * @return
 */
__device__ __host__ int* concat(int *arr1,int arr1Length,int *arr2,int arr2Length);


/**
 *
 * @param numArrays
 * @param numTotalElements
 * @param arr
 * @param lengths
 * @return
 */
__device__ __host__ int* concat(int  numArrays,int numTotalElements,int **arr,int *lengths);


/**
 * Get the length per slice of the
 * given shape and the dimension
 * @param rank the rank of the shape
 * @param shape the shape of to get
 * the length per slice for
 * @param dimension the dimension to
 * get the length per slice for
 * @param dimensionLength the length of the dimension array
 * @return the length per slice of the given shape
 * along the given dimension
 */
__device__ __host__ int lengthPerSlice(int rank,int *shape,int *dimension,int dimensionLength);

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
__device__ __host__ int sliceOffsetForTensor(int rank,int index, int *shape, int *tensorShape,int tensorShapeLength,int *dimension,int dimensionLength);

/**
 * Computes the offset for accessing
 * a global element given the shape information
 * and the offset to be read.
 */
__device__ int tadOffset(int *xInfo,int offset);

/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
__device__ __host__ int offset(int index,int rank,nd4j::shape::ShapeInformation *info,int *dimension,int dimensionLength);


/**
 * TADPermuteInfo is for intermediate information
 * needed for computing tensor along dimension.
 *
 *
 */
typedef struct  {
	int *tensorShape;
	int xRank;
	int *reverseDimensions;
	int *rangeRet;
	int removeLength;
	int *remove;
	int *zeroDimension;
	int *newPermuteDims;
	int *permutedShape;
	int *permutedStrides;
	int tensorShapeLength;
	int tensorShapeProd;
} TADPermuteInfo;


/**
 * Given the shape information and dimensions
 * returns common information
 * needed for tensor along dimension
 * calculations
 */
__device__ __host__ TADPermuteInfo tadInfo(int *xShapeInfo,int *dimension,int dimensionLength);



/**
 * Frees the permute information
 * @param info the info to free
 */
__host__ __device__ void freePermuteInfo(TADPermuteInfo info);


/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
__device__ __host__ int tensorsAlongDimension(volatile int rank,volatile int length,volatile int *shape,int *dimension,int dimensionLength);

/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
__device__ __host__ int tensorsAlongDimension(int *shapeInfo,int *dimension,int dimensionLength);



/**
 *
 * @param info
 * @param dimension
 * @param dimensionLength
 * @return
 */
__device__ __host__ int tensorsAlongDimension(TADPermuteInfo info,int *dimension,int dimensionLength);

/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
__device__ __host__ int offset(int index,int *xShapeInfo,int *dimension,int dimensionLength,TADPermuteInfo info);

/**
 * Returns the tensor along dimension
 * for the given block index
 * @param blockSize
 * @param blockIdx
 * @param i
 * @return
 */
__device__ __host__ int tadForBlockIndex(int blockSize,int blockIdx,int i);



/**
 * Computes the number of tads per block
 *
 */
__device__ __host__ int tadsPerBlock(int blockSize,int tads);


/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
__device__ __host__ int * toShapeBuffer(nd4j::shape::ShapeInformation *info);
}
}

#endif /* TAD_H_ */
