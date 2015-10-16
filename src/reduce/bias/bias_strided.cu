#include <reduce.h>

__device__ double merge(double old,double opOutput,double *extraParams) {
	return opOutput + old;
}

__device__ double update(double old,double opOutput,double *extraParams) {
	return opOutput + old;
}


/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
 */
__device__ double op(double d1,double d2,double *extraParams) {
	return op(d1,extraParams);
}
//an op for the kernel
__device__ double op(double d1,double *extraParams) {
	double mean = extraParams[1];
	double curr = (d1 - mean);
	return  curr;

}

//post process result (for things like means etc)
__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
	return reduction;
}

extern "C"
__global__ void bias_strided_double(
		int n
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


/**
 An op on the device
 @param d1 the first operator
 @param d2 the second operator
 */
__device__ float op(float d1,float d2,float *extraParams) {
	return op(d1,extraParams);
}
//an op for the kernel
__device__ float op(float d1,float *extraParams) {
	float mean = extraParams[1];
	float curr = (d1 - mean);
	return  curr;

}

//post process result (for things like means etc)
__device__ float postProcess(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
	return reduction;
}

extern "C"
__global__ void bias_strided_float(	int n
		,float *dx
		,int *xVectorInfo
		,float *extraParams
		,float *result,
		int *resultVectorInfo
		,int *gpuInformation,
		int *problemDefinition) {
	transform<float>(n,dx,xVectorInfo,extraParams,result,resultVectorInfo,gpuInformation,problemDefinition);
}


