/*
 * mathutil.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef MATHUTIL_H_
#define MATHUTIL_H_


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




#endif /* MATHUTIL_H_ */
