#include <reduce.h>


template<> __device__ double update<double>(double old,double opOutput,double *extraParams) {
	return opOutput + old;
}


template<> __device__ double merge<double>(double old,double opOutput,double *extraParams) {
	return opOutput + old;
}

template<> __device__ double op<double>(double d1,double *extraParams) {
	return d1;
}


template<> __device__ double postProcess<double>(double reduction,int n,int xOffset,double *dx,int incx,double *params,double *result) {
	return reduction;
}



template<> __device__ float update<float>(float old,float opOutput,float *extraParams) {
	return opOutput + old;
}


template<> __device__ float merge<float>(float old,float opOutput,float *extraParams) {
	return opOutput + old;
}

template<> __device__ float op<float>(float d1,float *extraParams) {
	return d1;
}


template<> __device__ float postProcess<float>(float reduction,int n,int xOffset,float *dx,int incx,float *params,float *result) {
	return reduction;
}





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
