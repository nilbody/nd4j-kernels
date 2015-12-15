#include <tad.h>
#include <helper_cuda.h>
#include <indexing.h>
#include <sharedmem.h>



/**
 * Given an linear index, element wise stride
 * and the length of each tad
 * map a linear index to a tad
 * @param i the index to map
 * @param the element wise stride for the tads
 * @param numElementsPerTad the number of elements
 * per tad
 */
__device__ __host__ int tadIndex(int i,int elementWiseStride,int numElementsPerTad) {
	return i / (numElementsPerTad * elementWiseStride);
}

/**
 * Map a tad to a
 * reduction index.
 * @param tadIndexForOriginal the original tad index for the
 * split up problem (eg: split is dimension 3 mapping to a 2,3 problem)
 * @param tadsForReduced the number of tads for the shrunk down problem (eg: 2,3)
 * @param tadsForOriginal the number of tads for the smaller problem (eg: 3)
 */
__device__ __host__ int reductionIndexForTad(int tadIndexForOriginal,int tadsForReduced,int tadsForOriginal) {
	if(tadIndexForOriginal == 0)
		return 0;
	return tadIndexForOriginal / (tadsForOriginal / tadsForReduced);
}

/**
 * Computes the number of tads
 * per reduce index for the
 * reduction tad.
 */
__device__ __host__ int tadsPerReduceIndex(int tadsForReduce,int tadsForOriginal) {
	return tadsForOriginal / tadsForReduce;
}

/**
 * Maps the given index
 * based
 * @param origingalTads the number of tads for a
 * the multiplied problem
 * @param numTads the number of tads for the
 * shrunken/multi dimension problem
 *
 */
__device__ __host__ int tadIndexForExpanded(int originalTads,int numTads,int idx) {
	return idx / tadsPerReduceIndex(numTads,originalTads);
}

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
		,int originalTadNum) {
	int tad = tadIndex(i,elementWiseStride,numElementsPerTad);
	return reductionIndexForTad(tad,tadNum,originalTadNum);
}

/**
 * This implements a collapsing tad reduction
 * based on different dimensions.
 *
 * The reason we need this is because of the fact that
 * there are certain dimension combinations (usually > 1)
 * that don't have an element wise stride.
 *
 * A way to bypass this problem is to expand the problem
 * in to a 1 dimension reduction problem
 * and then collapsing the results in to the equivalent
 * shape of the multi dimension problem.
 *
 * An example problem would be an array of:
 * linspace(1,24,24).reshape(2,2,3,2)
 *
 * The tad for reduction:
 * 2,3 doesn't have an element wise stride.
 *
 * However, the tad for reduction:
 * 3 does
 *
 * What we can exploit here is the ability
 * to reshape problems of multiple dimensions
 *
 * in to equivalent expanded problems based on smaller tads
 * eg:
 * multiple reductions for each dimension along dimension 3
 * followed by collapsing the problem in to an equivalent state
 * as if we had specified 2,3 for the dimensions instead.
 *
 * This gives us a way of executing an element wise stride based
 * algorithm  that is executable on the gpu.
 *
 * For the GPU, we force each block to process a  tad
 * at the singular dimension level. Eg: dimension 3
 *
 * So for example along dimension 3 of the 2,2,3,2
 * array we have 12 tensors along dimension.
 *
 * We then map those 12 tads to a reduction index.
 *
 * A reduction index is the equivalent value
 * in teh result as if we had specified the reduction dimensions
 * to be 2,3 instead.
 *
 * For example, if we have 12 tads for dimension 3
 * we will only have 4 for dimensions 2,3
 *
 * The goal will be then to generate the equivalent results
 * using dimension 3 but collapsing the results according to
 * the dimension 2,3 space (remember: the reason we are doing this mapping
 * is because we are trying to map the multi dimensional problem on to
 * a problem that allows us to solve it via element wise stride)
 *
 *
 * An example mapping relative to a gpu block is as follows:
 * ([[[[  1.,   2.],
         [  3.,   4.],
         [  5.,   6.]],

        [[  7.,   8.],
         [  9.,  10.],
         [ 11.,  12.]]],


       [[[ 13.,  14.],
         [ 15.,  16.],
         [ 17.,  18.]],

        [[ 19.,  20.],
         [ 21.,  22.],
         [ 23.,  24.]]]])



 * Along dimension 3 we will have tads of length 2
 * and 4 reduction indexes we need to map for the
 * 2,3 dimension problem.
 *
 *
 * The first reduction index will map to the first 3 tads of length 2
 * The next reduction index will map to the next 3, etc.
 *
 * We then process a reduction index per block on the gpu.
 * If any gpu block index is > the number of
 * reduction indexes we skip it.
 *
 * Note here we did this implementation because of
 * race conditions on the block and shared memory.
 *
 * This way of mapping allows us to avoid race conditions.
 *
 * @param data the data to process
 * @param result the result vector
 * @param initialValue the initial value for the reductino
 * @param elementsPerTad the elements per tad
 * for the expanded tad (eg: the one being collapsed from)
 * @param numTads the number of tads for the final result
 * @param n the number of elements in the buffer total
 * @param elementWiseStride the element wise stride
 * we use for the singular dimensions for each tad
 * @param numOriginalTads the number of original tads for the expanded version (eg: we are doing
 * reduction mapping a single dimension problem that allows for an element wise stride on to a multi
 * index problem)
 * @param sharedMemorySize the shared memory size we specified for launching the kernel - this is used for figuring out
 * how many elements are possible for the shared memory buffer for initializing the values to be default
 * @param xShapeInfo the shape information for the buffer - for more information on this see tad.h
 * @param dimension the dimension for the problem on the smaller scale (eg: the expanded version of the problem)
 * @param dimensionLength the length of the number of dimensions
 *
 */
template <typename T>
__device__ void collapseTad(
		T *data
		,T *result
		,T initialValue
		,int elementsPerTad
		,int numTads
		,int n
		,int elementWiseStride
		,int numOriginalTads,int sharedMemorySize,
		int *xShapeInfo
		,int *dimension,int dimensionLength) {
	SharedMemory<T> val;
	volatile T *sPartials = val.getPointer();
	int tid = threadIdx.x;
	//intialize te values
	int numItems = sharedMemorySize / sizeof(T);

	for (int i = tid; i < numItems; i += blockDim.x) {
		sPartials[i] = initialValue;
	}
	__syncthreads();

	//each block processes a reduction index
	if(blockIdx.x >= numTads)
		return;


	__shared__ TADPermuteInfo xTadInfo;
	if(tid == 0) {
		xTadInfo  = tadInfo(xShapeInfo,dimension,dimensionLength);
	}

	__syncthreads();

	/**
	 * Reverse engineer which tads belong to a particular
	 * reduction index.
	 *
	 * Each tad should be handled by a thread.
	 *
	 * Combine them all in the block at the end.
	 *
	 *
	 */

	//number of tads per reduce index
	int tadsPerReduceIndex2 = tadsPerReduceIndex(numTads,numOriginalTads);
	//each thread does a tad
	if(tid >= tadsPerReduceIndex2)
		return;



	/**
	 * Need to ensure we stay in bounds on each block -
	 * we need to compute the proper tads for each block and
	 * do bounds checking on each thread.
	 *
	 * This is to ensure that each thread processes
	 * a unique tad at most once.
	 *
	 *
	 */
	/**
	 * NEXT PART HERE
	 */

	/**
	 * Now WRT the thread id
	 * we want to iterate through a tad
	 * on each thread using the element wise stride
	 * and num elements per tad to compute a reduce
	 * for the tad. We then reduce in shared memory
	 * setting the item in the shared memory space
	 * and aggregate all of thh partial results
	 * on thread 0 aggregating the final results
	 * on the block resulting in one global write.
	 */
	//compute the offset for the tad for this thread
	//iterating via element wise stride
	//note here blockidx.x + tid is the tad we want
	int tadForThread = tid + blockIdx.x * tadsPerReduceIndex2;
	int offsetForBlock = offset(tadForThread,xShapeInfo,dimension,dimensionLength,xTadInfo);

	for(int i = 0; i < elementsPerTad; offsetForBlock += elementWiseStride,i++) {
		sPartials[tid] += data[offsetForBlock];
		__syncthreads();
	}



	if(tid == 0 && blockIdx.x < numTads) {
		//start at 1 so we don't count the first entry twice
		for(int i = 1; i < numTads; i++) {
			sPartials[0] += sPartials[i];
			__syncthreads();
		}

		result[blockIdx.x] = sPartials[0];
		freePermuteInfo(xTadInfo);
	}
}


 __global__ void collapseTad_float(
 		float *data
 		,float *result
 		,float initialValue
 		,int elementsPerTad
 		,int numTads
 		,int n
 		,int elementWiseStride
 		,int numOriginalTads,int sharedMemorySize,
 		int *xShapeInfo
 		,int *dimension,int dimensionLength) {
	 collapseTad<float>(
			 data,
			 result,
			 initialValue,
			 elementsPerTad,
			 numTads,
			 n,
			 elementWiseStride,
			 numOriginalTads,
			 sharedMemorySize,
			 xShapeInfo,
			 dimension,
			 dimensionLength);

 }


 __global__ void collapseTad_double(
 		double *data
 		,double *result
 		,double initialValue
 		,int elementsPerTad
 		,int numTads
 		,int n
 		,int elementWiseStride
 		,int numOriginalTads,int sharedMemorySize,
 		int *xShapeInfo
 		,int *dimension,int dimensionLength) {
	 collapseTad<double>(
			 data,
			 result
			 ,initialValue
			 ,elementsPerTad,
			 numTads,
			 n,
			 elementWiseStride,
			 numOriginalTads,
			 sharedMemorySize,
			 xShapeInfo,
			 dimension,
			 dimensionLength);

 }

