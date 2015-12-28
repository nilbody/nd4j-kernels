
#include <math.h>
#include <stdio.h>

#include <sharedmem.h>
#include <tad.h>
#include <indexing.h>

namespace nd4j {
namespace functions {
namespace broadcast {



template <typename T>
class BaseBroadcast : public Broadcast<T> {

public:
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


		int xElementWiseStride = elementWiseStride(xShapeInfo);
		int xOffset = offset(xShapeInfo);
		int yElementWiseStride = elementWiseStride(yShapeInfo);
		int yOffset = offset(yShapeInfo);



		//length for the tad
		int yLength = length(yShapeInfo);
		//length for the tad
		int xLength  = length(xShapeInfo);

		int resultLength = length(resultShapeInfo);
		for (int i = blockIdx.x * blockDim.x + threadIdx.x;
				i < resultLength;
				i += blockDim.x * gridDim.x) {
			int yOffset2 = yOffset + ((i / xElementWiseStride)% yLength) * yElementWiseStride;
			if(i < resultLength)
				result[i] = op(x[i],y[yOffset2]);

		}

	}

};


}
}
}
