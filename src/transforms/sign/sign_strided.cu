#include <transform.h>


template<> __device__ double op<double>(double d1,double *params) {
	return (d1 > 0) - (d1 < 0);
}

extern "C"
__global__ void sign_strided_double(int n,int idx,double *dy,int incy,double *params,double *result,int blockSize) {
	transform<double>(n,idx,dy,incy,params,result,blockSize);

}


template<> __device__ float op<float>(float d1,float *params) {
	return (d1 > 0) - (d1 < 0);
}

extern "C"
__global__ void sign_strided_float(int n,int idx,float *dy,int incy,float *params,float *result,int blockSize) {
	transform<float>(n,idx,dy,incy,params,result,blockSize);

}
