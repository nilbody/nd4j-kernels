/*
 * shape.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SHAPE_H_
#define SHAPE_H_


namespace nd4j {
namespace shape {
/**
 * Shape information approximating
 * the information on an ndarray
 */
typedef struct {
	int *shape;
	int *stride;
	char order;
	int rank;
	int offset;
	int elementWiseStride;
} ShapeInformation;


/**
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
__device__ __host__ ShapeInformation *shapeCopy(ShapeInformation *toCopy);

/**
 *
 * @param length
 * @param shape
 * @param rearrange
 * @return
 */
__device__ __host__ int* doPermuteSwap(int length,int  *shape, int *rearrange);



/**
 * Get the ordering for the device
 * @param length
 * @param shape
 * @param stride
 * @param elementStride
 * @return
 */
__device__ __host__ char getOrder(int length ,int *shape,int *stride,int elementStride);

/**
 * Ensure that every value in the re arrange
 * array is unique
 * @param arr
 * @param shape
 * @param arrLength
 * @param shapeLength
 * @return
 */
__device__ __host__ int checkArrangeArray(int *arr,int *shape,int arrLength,int shapeLength);

/**
 * Permute the shape information
 * @param info the shape information to permute
 * @param rearrange the order to re arrange
 * @param rank the rank of the rearrange array
 */
__device__ __host__ void permute(ShapeInformation **info,int *rearrange,int rank);

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */
__device__ __host__ int isVector(int *shape,int rank);


/**
 * Returns the shape portion of an information
 * buffer
 */
__device__ __host__ int * shapeOf(int *buffer);





/**
 * Return a copy of a buffer.
 * This buffer allocates memory
 * that must be freed elsewhere.
 */
__device__ __host__ int *copyOf(int length,int *toCopy);

/**
 * Permute the given strides
 * in the given rearrange order
 * @param toPermute the buffer to permute
 * @param shapeRank the length of the buffer to permute
 * @param rearrange the rearrange order (must be 0 based indexes
 * and all must be filled in)
 * @return the rearranged array
 */
__device__ __host__ int * permutedStrides(int *toPermute,int shapeRank,int *rearrange);




/**
 * Return the slice (shape + 1 in pointer arithmetic)
 * @param shape the shape to take the slice of
 * @return the shape array - the first entry
 */
__device__ __host__ int *slice(int *shape);

/**
 * Returns the length of the
 * shape information buffer:
 * rank * 2 + 3
 * @param rank the rank to get the shape
 * info length for
 * @return rank * 2 + 4
 */
__device__ __host__ int shapeInfoLength(int rank);


/**
 * Returns the rank portion of
 * an information buffer
 */
__device__ __host__ int rank(int *buffer);



/**
 * Converts a raw int buffer of the layout:
 * rank
 * shape
 * stride
 * offset
 * elementWiseStride
 *
 * where shape and stride are both straight int pointers
 */
__device__ __host__ ShapeInformation* infoFromBuffer(int *buffer);

/**
 * Returns the stride portion of an information
 * buffer
 */
__device__ __host__ int *stride(int *buffer);


/**
 * Compute the length of the given shape
 */
__device__ __host__ int length(int *shapeInfo);

/***
 * Returns the offset portion of an information buffer
 */
__device__ __host__ int offset(int *buffer);


/**
 * Returns the ordering
 * for this shape information buffer
 */
__device__ __host__ char order(int *buffer);


/**
 * Returns the element wise stride for this information
 * buffer
 */
__device__ __host__ int elementWiseStride(int *buffer);

/**
 * Returns whether
 * the given shape info buffer
 * represents a scalar shape
 */
__device__ __host__ int isScalar(int *info);




/**
 * Returns whether
 * the given shape information
 * represents a scalar
 * shape or not
 */
__device__ __host__ int isScalar(volatile ShapeInformation *info);



}
}




#endif /* SHAPE_H_ */
