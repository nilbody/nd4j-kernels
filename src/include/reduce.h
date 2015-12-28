//an op for the kernel
namespace functions {
namespace reduce {


template <typename T>
class ReduceFunction {
public:
	/**
	 * Op with 1 parameter
	 * @param d1
	 * @param extraParams
	 * @return
	 */
	virtual __device__ T op(T d1,T *extraParams);

	//calculate an update of the reduce operation
	/**
	 * Op with 2 parameters
	 * @param old
	 * @param opOutput
	 * @param extraParams
	 * @return
	 */
	virtual __device__ T update(T old,T opOutput,T *extraParams);

	/**
	 * Merge 2 results
	 * @param f1
	 * @param f2
	 * @param extraParams
	 * @return
	 */
	//invoked when combining two kernels
	virtual __device__ T merge(T f1, T f2,T *extraParams);
	/**
	 * Op with 2 parameters
	 * @param d1
	 * @param d2
	 * @param extraParams
	 * @return
	 */
	//post process result (for things like means etc)
	virtual __device__ T op(T d1,T d2,T *extraParams);
	/**
	 * The actual reduction algorithm.
	 * @param n
	 * @param dx
	 * @param xShapeInfo
	 * @param extraParams
	 * @param result
	 * @param resultShapeInfo
	 * @param gpuInformation
	 * @param dimension
	 * @param dimensionLength
	 * @param postProcessOrNot
	 */
	virtual __device__ void transform(
			int n
			,T *dx
			,int *xShapeInfo
			,T *extraParams
			,T *result,
			int *resultShapeInfo
			,int *gpuInformation,
			int *dimension,
			int dimensionLength,int postProcessOrNot);

	/**
	 *
	 * @param reduction
	 * @param n
	 * @param xOffset
	 * @param dx
	 * @param incx
	 * @param extraParams
	 * @param result
	 * @return
	 */
	virtual __host__ __device__ T postProcess(T reduction,int n,int xOffset,T *dx,int incx,T *extraParams,T *result);
	virtual ~ReduceFunction();

};



template<typename T>
__device__ void aggregatePartials(T **sPartialsRef,int tid,T *extraParams);

template<typename T>
__device__ T doBlock(int n,T *sPartials,T *dx,int xOffset,int incx,T *extraParams);

template <typename T>
__device__ void initializeShared(T *extraParams,T** sPartials,int sMemSize);


/**
 * @param n n is the number of
 *        elements to loop through
 * @param dx the data to operate on
 * @param xVectorInfo the meta data for the vector:
 *                              0 is the offset
 *                              1 is the increment/stride
 *                              2 is the real length of the buffer (n and dx.length won't always be the same)
 *                              3 is the element wise stride for the buffer
 *                              4 is the number of elements it takes to get to the next row/column/tensor
 * @param gpuInformation
 *                              0 is the block size
 *                              1 is the grid size
 *                              2 is the shared memory size
 * @param problemDefinition
 *                          0 is the number of elements per vector
 *                          1 is the number of vectors
 */
template<typename T>
__device__ void transform(
		int n
		,T *dx
		,int *xShapeInfo
		,T *extraParams
		,T *result,
		int *resultShapeInfo
		,int *gpuInformation,
		int *dimension,
		int dimensionLength,int postProcessOrNot);

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
		,T *extraParams,
		int numOriginalTads,
		int sharedMemorySize,
		int *xShapeInfo
		,int *resultShapeInfo
		,int *dimension,int dimensionLength);

}
}












