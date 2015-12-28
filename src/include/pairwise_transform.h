/*
 * pairwise_transform.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef PAIRWISE_TRANSFORM_H_
#define PAIRWISE_TRANSFORM_H_


namespace nd4j {
namespace functions {
namespace pairwise_transforms {


template <typename T>
class PairWiseTransform {

	virtual __device__ T op(T d1,T d2,T *params);
	virtual __device__ T op(T d1,T *params);


	virtual __device__ void transform(
			int n,
			int xOffset,
			int yOffset,
			int resultOffset,
			T *dx,
			T *dy,
			int incx,
			int incy,
			T *params,
			T *result,int incz,int blockSize);

	virtual ~PairWiseTransform();

};


}
}
}


#endif /* PAIRWISE_TRANSFORM_H_ */
