//Original credit: https://llpanorama.wordpress.com/2008/06/11/threads-and-blocks-and-grids-oh-my/

#define MAX_THREADS_PER_BLOCK 1024

namespace nd4j {
namespace indexing {
/**

 * Returns the number of elements per thread
 */
__device__ int numElementsPerThread(int N);

/**
 * Returns the block starting index
 */
__device__ int blockStartingIndex(int N);


/**
 * Returns the thread starting index
 */
__device__ int threadStartingIndex(int N,int stride,int offset);


/**
 * Returns the thread ending index
 */
__device__ int threadEndingIndex(int N,int stride,int offset);




/**
 * Indexing information
 * for bounds checking
 */
typedef struct {
	int numElementsPerThread;
	int blockStartingIndex;
	int startingThreadIndex;
	int endingThreadIndex;

} CurrentIndexing;


/**
 * Returns indexing information
 * for the current kernel invocation
 */
__device__ CurrentIndexing* currentIndex(int N,int offset,int stride);
}
}


