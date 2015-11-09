#include <reduce.h>

template<> __device__ double merge<double>(double old,double opOutput,double *extraParams) {
	return opOutput + old;
}

template<> __device__ double update<double>(double old,double opOutput,double *extraParams) {
	return opOutput + old;
}


template<> __device__ double op<double>(double d1,double *extraParams) {
	return pow(d1,2);
}


template<> __device__ double postProcess<double>(double reduction,int n,int xOffset,double *dx,int incx,double *params,double *result) {
	return sqrt(reduction);
}





template<> __device__ float merge<float>(float old,float opOutput,float *extraParams) {
	return opOutput + old;
}

template<> __device__ float update<float>(float old,float opOutput,float *extraParams) {
	return opOutput + old;
}


template<> __device__ float op<float>(float d1,float *extraParams) {
	return powf(d1,2);
}


template<> __device__ float postProcess<float>(float reduction,int n,int xOffset,float *dx,int incx,float *params,float *result) {
	return sqrtf(reduction);
}







