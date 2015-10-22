/*
 * tad.h
 *
 *  Created on: Oct 21, 2015
 *      Author: agibsonccc
 */

#ifndef TAD_H_
#define TAD_H_
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
 * Returns the prod of the data
 * up to the given length
 */
__device__ __host__ int prod(int *data,int length) {
	int prod = 1;
	for(int i = 0; i < length; i++) {
		prod *= data[i];
	}

	return prod;
}

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
__device__ __host__ int*  removeIndex(int *data,int *indexes,int dataLength,int indexesLength) {
	int *ret = (int *) malloc(dataLength - indexesLength);
	int count = 0;
	for(int i = 0; i < dataLength; i++) {
		int contains = 0;
		for(int j = 0; j < indexesLength; j++) {
			if(i == indexes[j]) {
				contains = 1;
				break;
			}
		}

		if(!contains)
			ret[count++] = data[i];
	}

	return ret;
}

/**
 * Generate an int buffer
 * up to the given length
 * at the specified increment
 *
 */
__device__ __host__ int* range(int from,int to,int increment) {
	int diff = abs(from - to);
	int retLength = diff / increment;
	int *ret = new int[diff / increment];
	if(diff / increment < 1)
		ret = new int[1];

	if (from < to) {
		int count = 0;
		for (int i = from; i < to; i += increment) {
			if (count >= retLength)
				break;
			ret[count++] = i;
		}
	} else if (from > to) {
		int count = 0;
		for (int i = from - 1; i >= to; i -= increment) {
			if (count >= retLength)
				break;
			ret[count++] = i;
		}
	}

	return ret;
}

/**
 * Range between from and two with an
 * increment of 1
 */
__device__ __host__ int* range(int from,int to) {
	return range(from,to,1);
}

/**
 * Keep the given indexes
 * in the data
 */
__device__ __host__ int* keep(int *data,int *index,int indexLength,int dataLength) {
	int *ret = (int *) malloc(indexLength * sizeof(int));
	int count = 0;
	for(int i = 0; i < dataLength; i++) {
		int contains = 0;
		for(int j = 0; j < indexLength; j++) {
			if(i == index[j]) {
				contains = 1;
				break;
			}
		}

		if(contains)
			ret[count++] = data[i];
	}
	return ret;
}


/**
 * Generate a reverse
 * copy of the data
 */
__device__ __host__ int* reverseCopy(int *data,int length) {
	if (length < 1)
		return data;

	int *copy = (int *) malloc(length * sizeof(int));
	for (int i = 0; i <= length / 2; i++) {
		int temp = data[i];
		copy[i] = data[length - i - 1];
		copy[length - i - 1] = temp;
	}
	return copy;
}

/**
 * Permutes the given dimensions
 */
__device__ __host__ int* doPermuteSwap(int length,int  *shape, int *rearrange) {
	int *ret = new int[length];
	for (int i = 0; i < length; i++) {
		ret[i] = shape[rearrange[i]];
	}
	return ret;
}

__device__ __host__ int checkArrangeArray(int *arr,int *shape,int arrLength,int shapeLength) {
	if(arrLength != shapeLength)
		return -1;
	for (int i = 0; i < arrLength; i++) {
		if (arr[i] >= arrLength || arr[i] < 0)
			return -1;
	}

	for (int i = 0; i < arrLength; i++) {
		for (int j = 0; j < arrLength; j++) {
			if (i != j && arr[i] == arr[j])
				return -1;
		}
	}

	return 1;
}

__device__ __host__ char getOrder(int length ,int *shape,int *stride,int elementStride) {
	int sd;
	int dim;
	int i;
	int cContiguous = 1;
	int isFortran = 1;

	sd = 1;
	for (i = length - 1; i >= 0; --i) {
		dim = shape[i];

		if (stride[i] != sd) {
			cContiguous = 0;
			break;
		}
		/* contiguous, if it got this far */
		if (dim == 0) {
			break;
		}
		sd *= dim;

	}


	/* check if fortran contiguous */
	sd = elementStride;
	for (i = 0; i < length; ++i) {
		dim = shape[i];
		if (stride[i] != sd) {
			isFortran = 0;
		}
		if (dim == 0) {
			break;
		}
		sd *= dim;

	}

	if(isFortran && cContiguous)
		return 'a';
	else if(isFortran && !cContiguous)
		return 'f';
	else if(!isFortran && !cContiguous)
		return 'c';
	else
		return 'c';

}


__device__ __host__ int* concat(int  numArrays,int numTotalElements,int **arr,int *lengths) {
	int *ret = (int *) malloc(numTotalElements * sizeof(int));
	int count = 0;
	for(int i = 0; i < numArrays; i++) {
		for(int j = 0; j < lengths[i]; j++) {
			ret[count++] = arr[i][j];

		}
	}

	return ret;
}



__device__ __host__ int lengthPerSlice(int rank,int *shape,int *dimension) {
	int *ret2 = removeIndex(shape,dimension,rank,rank);
	int ret = prod(ret2,rank);
	free(ret2);
	return ret;
}

/**
 * calculates the offset for a tensor
 * @param index
 * @param arr
 * @param tensorShape
 * @return
 */
__device__ __host__ int sliceOffsetForTensor(int rank,int index, int *shape, int *tensorShape) {
	int tensorLength = prod(tensorShape,rank);
	int offset = index * tensorLength / lengthPerSlice(rank,shape,tensorShape);
	return offset;
}






__device__ __host__ void permute(ShapeInformation *info,int *rearrange,int rank) {
	checkArrangeArray(rearrange,info->shape,rank,rank);
	int *newShape = doPermuteSwap(rank,info->shape,rearrange);
	int *newStride = doPermuteSwap(rank,info->stride,rearrange);
	char order = getOrder(rank,info->shape,info ->stride,info->elementWiseStride);
	//free the old shape and strides
	free(info->shape);
	free(info->stride);
	info->shape = newShape;
	info->stride = newStride;
	info->order = order;

}


__device__ __host__ int *slice(int *shape,int rank) {
	int *ret = (int *) malloc(rank - 1 * sizeof(int));
	for(int i = 0; i < rank - 1; i++) {
		ret[i] = shape[i + 1];
	}

	return ret;
}


__device__ __host__ int offset(int index,int rank,ShapeInformation *info,int *dimension) {
	int  *tensorShape = keep(info->shape,dimension,rank,rank);
	int  *reverseDimensions = reverseCopy(dimension,rank);
	int *rangeRet = range(0, rank);
	int  *remove = removeIndex(rangeRet, dimension,rank,rank);
	free(rangeRet);
	int **pointers = (int **)malloc(2 * sizeof(int *));
	pointers[0] = remove;
	pointers[1] = reverseDimensions;

	//int  numArrays,int numTotalElements,int **arr,int *lengths

	int *lengths = (int *) malloc(2 * sizeof(int));
	for(int i = 0; i < 2; i++)
		lengths[i] = rank;

	int *newPermuteDims = concat(2,rank * rank,pointers,lengths);
	//__device__ void permute(ShapeInformation *info,int *rearrange,int rank) {
	permute(info,newPermuteDims,rank);
	int *permuted = info->shape;
	int sliceIdx = sliceOffsetForTensor(rank,index, permuted, tensorShape);

	int  *ret2 = slice(info->shape,rank);
	int ret2Length = prod(ret2,rank - 1);
	int ret2Rank = rank - 1;
	if(rank == rank && prod(tensorShape,rank) == ret2Length)
		return info->offset;

	int length = prod(tensorShape,rank);
	int tensorLength = length;
	//__device__ int lengthPerSlice(int rank,int *shape,int *dimension) {
	int offset = index * tensorLength / lengthPerSlice(ret2Rank,ret2,ret2);

	if(sliceIdx == 0 && length == lengthPerSlice(rank - 1,ret2,permuted)) {
		return offset;
	}

	if(length == lengthPerSlice(rank - 1,ret2,ret2)) {
		offset -= ret2[0] * (offset / ret2[0]);
		int *oldRet2 = ret2;
		ret2 = slice(ret2,ret2Rank);
		free(oldRet2);
		ret2Rank--;
		return offset;
	}

	while(ret2Length > length) {
		sliceIdx = sliceOffsetForTensor(rank,index, ret2, tensorShape);
		sliceIdx -= ret2[0] * (sliceIdx / ret2[0]);
		int *oldRet2 = ret2;
		ret2 = slice(info->shape,ret2Rank);
		free(oldRet2);
		ret2Rank--;
		length -= prod(ret2,ret2Rank);
	}

	free(pointers);
	free(ret2);
	free(tensorShape);
	free(reverseDimensions);
	free(rangeRet);
	free(remove);


	return  offset;
}









#endif /* TAD_H_ */
