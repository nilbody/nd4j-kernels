#include <reduce.h>


__device__ double update(double old,double opOutput,double *extraParams) {
	return opOutput + old;
}


__device__ double merge(double old,double opOutput,double *extraParams) {
	return opOutput + old;
}

__device__ double op(double d1,double *extraParams) {
	return d1;
}


__device__ double postProcess(double reduction,int n,int xOffset,double *dx,int incx,double *params,double *result) {
	return reduction;
}


extern "C"
__global__ void sum_strided_double(	int n
		,double *dx
		,int *xVectorInfo
		,double *extraParams
		,double *result,
		int *resultVectorInfo
		,int *gpuInformation,
		int *problemDefinition) {
	transform<double>(n,dx,xVectorInfo,extraParams,result,resultVectorInfo,gpuInformation,problemDefinition);

}


extern "C"
__global__ void sum_strided_float(	int n
		,float *dx
		,int *xVectorInfo
		,float *extraParams
		,float *result,
		int *resultVectorInfo
		,int *gpuInformation,
		int *problemDefinition) {
	transform<float>(n,dx,xVectorInfo,extraParams,result,resultVectorInfo,gpuInformation,problemDefinition);

}
