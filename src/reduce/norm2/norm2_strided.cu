#include <reduce.h>

__device__ double merge(double old,double opOutput,double *extraParams) {
	return opOutput + old;
}

__device__ double update(double old,double opOutput,double *extraParams) {
	return opOutput + old;
}


__device__ double op(double d1,double *extraParams) {
	return pow(d1,2);
}


__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *params,double *result) {
	return sqrt(reduction);
}
extern "C"
__global__ void norm2_strided_double(	int n
		,double *dx
		,int *xVectorInfo
		,double *extraParams
		,double *result,
		int *resultVectorInfo
		,int *gpuInformation,
		int *problemDefinition) {
	transform<double>(n,dx,xVectorInfo,extraParams,result,resultVectorInfo,gpuInformation,problemDefinition);
}


__device__ float merge(float old,float opOutput,float *extraParams) {
	return opOutput + old;
}

__device__ float update(float old,float opOutput,float *extraParams) {
	return opOutput + old;
}


__device__ float op(float d1,float *extraParams) {
	return powf(d1,2);
}


__device__ float postProcess(float reduction,int n,int xOffset,float *dx,int incx,float *params,float *result) {
	return sqrtf(reduction);
}

extern "C"
__global__ void norm2_strided_float(	int n
		,float *dx
		,int *xVectorInfo
		,float *extraParams
		,float *result,
		int *resultVectorInfo
		,int *gpuInformation,
		int *problemDefinition) {
	transform<float>(n,dx,xVectorInfo,extraParams,result,resultVectorInfo,gpuInformation,problemDefinition);
}




