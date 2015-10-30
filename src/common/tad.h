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
}


__device__ __host__ int isScalar(ShapeInformation *info) {
	if(info->rank > 2)
		return 0;
	if(info->rank == 1)
		return info->shape[0] == 1;
	else if(info->rank == 2) {
		return info->shape[0] == 1 && info->shape[1] == 1;
	}
	return 0;
}

/**
 * @param toCopy the shape to copy
 * @return a copy of the original struct
 */
__device__ __host__ ShapeInformation *shapeCopy(ShapeInformation *toCopy) {
	ShapeInformation *copy = (ShapeInformation *) malloc(sizeof(ShapeInformation));
	copy->shape = (int *) malloc(sizeof(int) * toCopy->rank);
	for(int i = 0; i < toCopy->rank; i++) {
		copy->shape[i] = toCopy->shape[i];
	}


	copy->stride = (int *) malloc(sizeof(int) * toCopy->rank);
	for(int i = 0; i < toCopy->rank; i++) {
		copy->stride[i] = toCopy->stride[i];
	}
	copy->order = toCopy->order;
	copy->rank = toCopy->rank;
	copy->offset = toCopy->offset;
	copy->elementWiseStride = toCopy->elementWiseStride;
	return copy;
}

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
__device__ __host__ void  removeIndex(int *data,int *indexes,int dataLength,int indexesLength,int **out) {
	int *ret = (int *) *out;
	int count = 0;
	for(int i = 0; i < dataLength; i++) {
		int contains = 0;
		for(int j = 0; j < indexesLength; j++) {
			if(i == indexes[j]) {
				contains = 1;
				break;
			}
		}

		if(!contains) {
			int currI = data[i];
			ret[count] = currI;
			count++;
		}
	}
}

/**
 * Returns a shape
 * forces the given length to be 2.
 * @param shape the shape to modify
 * @param dimension the dimension (row or column)
 * for the shape to be returned as
 * @return the new shape
 */
__device__ __host__ int*  ensureVectorShape(int *shape,int dimension) {
	int *ret = new int[2];
	if(dimension == 0) {
		ret[0] = 1;
		ret[1] = shape[0];
	}
	else {
		ret[0] = shape[0];
		ret[1] = 1;
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
	int *ret = (int *) malloc((indexLength) * sizeof(int));
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



__device__ __host__ int lengthPerSlice(int rank,int *shape,int *dimension,int dimensionLength) {
	int *ret2 = (int *) malloc((rank - dimensionLength) * sizeof(int));
	removeIndex(shape,dimension,rank,dimensionLength,&ret2);
	int length = rank - dimensionLength;
	int ret = prod(ret2,length);
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
__device__ __host__ int sliceOffsetForTensor(int rank,int index, int *shape, int *tensorShape,int tensorShapeLength,int *dimension,int dimensionLength) {
	int tensorLength = prod(tensorShape,tensorShapeLength);
	int lengthPerSlice2 = lengthPerSlice(rank,shape,dimension,dimensionLength);
	int offset = index * tensorLength / lengthPerSlice2;
	return offset;
}





/**
 *
 */
__device__ __host__ void permute(ShapeInformation **info,int *rearrange,int rank) {
	ShapeInformation *infoDeref = (ShapeInformation *) *info;
	checkArrangeArray(rearrange,infoDeref->shape,rank,rank);
	int *newShape = doPermuteSwap(rank,infoDeref->shape,rearrange);
	int *newStride = doPermuteSwap(rank,infoDeref->stride,rearrange);
	char order = getOrder(rank,infoDeref->shape,infoDeref ->stride,infoDeref->elementWiseStride);
	//free the old shape and strides
	free(infoDeref->shape);
	free(infoDeref->stride);
	infoDeref->shape = newShape;
	infoDeref->stride = newStride;
	infoDeref->order = order;

}

/**
 *
 */
__device__ __host__ int *slice(int *shape,int rank) {
	int *ret = (int *) malloc((rank - 1) * sizeof(int));
	for(int i = 0; i < rank - 1; i++) {
		ret[i] = shape[i + 1];
	}

	return ret;
}

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
__device__ __host__ ShapeInformation* infoFromBuffer(int *buffer) {
	ShapeInformation *info = (ShapeInformation *) malloc(sizeof(ShapeInformation));
	int length = buffer[0] * 2 + 4;
	int rank = buffer[0];

	//start after rank
	info->shape = buffer + 1;
	info->stride = buffer + (1 + rank);
	info->rank = rank;
	info->offset = buffer[length - 3];
	info->elementWiseStride = buffer[length - 2];
	int *stride = buffer + 1 + rank;
	info->stride = stride;
	info->order = (char) buffer[length - 1];

	return info;
}


/**
 * Computes the number
 * of tensors along
 * a given dimension
 */
__device__ __host__ int tensorsAlongDimension(int rank,int length,int *shape,int *dimension,int dimensionLength) {
	int *tensorShape = keep(shape,dimension,rank,dimensionLength);
	int ret = length / prod(tensorShape,dimensionLength);
	free(tensorShape);
	return ret;
}

/**
 * Returns whether the
 * given shape is a vector or not
 * @param shape the shape of the array
 * @param rank the rank of the shape
 */
__device__ __host__ int isVector(int *shape,int rank) {
	if(rank > 2)
		return 0;
	else if(rank <= 2) {
		if(shape[0] == 1 || shape[1] == 1)
			return 1;
	}
	return 0;
}

/**
 * Returns the length of the
 * shape information buffer:
 * rank * 2 + 3
 * @param rank the rank to get the shape
 * info length for
 * @return rank * 2 + 4
 */
__device__ __host__ int shapeInfoLength(int rank) {
	return rank * 2 + 4;
}

/**
 * Computes the tensor along dimension
 * offset
 * @param index the index to get the offset for the tad for
 * @param rank the rank of the shapes and strides
 * @param info the shape information to use for tad
 * @param dimension the dimensions to use for computing the tensor along dimensions
 */
__device__ __host__ int offset(int index,int rank,ShapeInformation *info,int *dimension,int dimensionLength) {
	int  *tensorShape = keep(info->shape,dimension,dimensionLength,rank);
	if(rank - dimensionLength <= 2) {
		int *newTensorShape = ensureVectorShape(tensorShape,dimension[0]);
		free(tensorShape);
		tensorShape = newTensorShape;
	}

	//change the value
	ShapeInformation *copy = shapeCopy(info);
	info = copy;

	int  *reverseDimensions = reverseCopy(dimension,dimensionLength);
	int *rangeRet = range(0, rank);
	int *remove = (int *) malloc((rank - dimensionLength) * sizeof(int));
	removeIndex(rangeRet, dimension,rank,dimensionLength,&remove);

	int *zeroDimension = (int *) malloc(1 * sizeof(int));
	zeroDimension[0] = 0;

	int removeLength = rank - dimensionLength;
	int **pointers = (int **)malloc(2 * sizeof(int *));
	pointers[0] = remove;
	pointers[1] = reverseDimensions;


	int *lengths = (int *) malloc(2 * sizeof(int));
	lengths[0] = removeLength;
	lengths[1] = dimensionLength;
	int *newPermuteDims = concat(2,removeLength + dimensionLength,pointers,lengths);
	//__device__ void permute(ShapeInformation *info,int *rearrange,int rank) {
	permute(&info,newPermuteDims,rank);

	int *permuted = info->shape;
	int *permutedStrides = info->stride;
	int tensorShapeLength = rank - removeLength;
	if(tensorShapeLength < 2)
		tensorShapeLength = 2;
	int sliceIdx = sliceOffsetForTensor(rank,index, permuted, tensorShape,tensorShapeLength,zeroDimension,1);

	//determine offset here

	int  *ret2 = slice(info->shape,rank);
	int ret2Length = prod(ret2,rank - 1);
	int ret2Rank = rank - 1;

	int retOffset = sliceIdx * permutedStrides[0];
	int tensorShapeProd = prod(tensorShape,tensorShapeLength);



	int length = prod(tensorShape,tensorShapeLength);
	int tensorLength = length;
	//__device__ int lengthPerSlice(int rank,int *shape,int *dimension) {
	int offset = index * tensorLength / lengthPerSlice(ret2Rank,ret2,zeroDimension,1);
	/**
	 * Need to do slice(offset) here
	 */
	if(sliceIdx == 0 && length == lengthPerSlice(ret2Rank,ret2,zeroDimension,1)) {
		/**
		 * NOTE STRIDE[1] HERE. WE DO THIS TO AVOID CREATING A NEW SLICE OBJECT.
		 */
		retOffset = info->offset + offset  * info->stride[1];
	}

	//determine offset here
	//note here offset doesn't change, just the shape
	//of the tad
	else if(length == lengthPerSlice(ret2Rank,ret2,zeroDimension,1)) {
		offset -= ret2[0] * (offset / ret2[0]);
		//set offset here
		ret2 = slice(ret2,ret2Rank);
		ret2Rank--;
		retOffset += info->stride[1] * offset;
	}


	else {
		while(ret2Length > length) {
			sliceIdx = sliceOffsetForTensor(rank,index, ret2, tensorShape,tensorShapeLength,zeroDimension,1);
			sliceIdx -= ret2[0] * (sliceIdx / ret2[0]);
			int *oldRet2 = ret2;
			//set offset
			ret2 = slice(info->shape,ret2Rank);
			free(oldRet2);
			ret2Rank--;
			//slice wise offsets are offset + i * majorStride()
			//dividing by the slice index will adjust the offset by a factor of sliceIndex
			retOffset /= sliceIdx;
			length -= prod(ret2,ret2Rank);

		}
	}
	/*
	free(pointers);
	free(ret2);
	 */
	free(reverseDimensions);
	free(rangeRet);
	free(remove);
	free(copy);

	//free the new pointer
	if(rank <= 2) {
		free(tensorShape);
	}

	return  retOffset;
}


/**
 * Returns a shape buffer
 * for the shape information metadata.
 */
__device__ __host__ int * toShapeBuffer(ShapeInformation *info) {
	int *ret = new int[shapeInfoLength(info->rank)];
	int count = 1;
	ret[0] = info->rank;
	for(int i = 0; i < info->rank; i++) {
		ret[count++] = info->shape[i];
	}
	for(int i = 0; i < info->rank; i++) {
		ret[count++] = info->stride[i];
	}

	ret[count++] = info->offset;
	ret[count++] = info->elementWiseStride;
	ret[count++] = info->order;


	return ret;
}





#endif /* TAD_H_ */
