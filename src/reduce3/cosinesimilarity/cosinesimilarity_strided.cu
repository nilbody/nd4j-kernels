#include <reduce3.h>


template<> __device__ double update<double>(double old,double opOutput,double *extraParams) {
	return old + opOutput;
}


/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
 */
template<> __device__ double op<double>(double d1,double d2,double *extraParams) {
	return d1 * d2;
}


//post process result (for things like means etc)
template<> __device__ double postProcess<double>(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
	return reduction / extraParams[1] / extraParams[2];
}


__global__ void cosinesimilarity_strided_double(int n, int xOffset,int yOffset,double *dx,double *dy,int incx,int incy,double *extraParams,double *result,int i,int blockSize) {
	transform_pair<double>(n,xOffset,yOffset,dx,dy,incx,incy,extraParams,result,i,blockSize);

}



template<> __device__ float update<float>(float old,float opOutput,float *extraParams) {
	return old + opOutput;
}


/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
 */
template<> __device__ float op<float>(float d1,float d2,float *extraParams) {
	return d1 * d2;
}


//post process result (for things like means etc)
template<> __device__ float postProcess<float>(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
	return reduction / extraParams[1] / extraParams[2];
}


__global__ void cosinesimilarity_strided_float(int n, int xOffset,int yOffset,float *dx,float *dy,int incx,int incy,float *extraParams,float *result,int i,int blockSize) {
	transform_pair<float>(n,xOffset,yOffset,dx,dy,incx,incy,extraParams,result,i,blockSize);

}



