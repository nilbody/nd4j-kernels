#include <transform.h>


template<> __device__ double op<double>(double d1,double *params) {
	return tanh(d1);
}


__global__ void tanh_strided_double(int n,int idx,double *dy,int incy,double *params,double *result,int blockSize) {
	transform<double>(n,idx,dy,incy,params,result,blockSize);

}

template<> __device__ float op<float>(float d1,float *params) {
	return tanhf(d1);
}


__global__ void tanh_strided_float(int n,int idx,float *dy,int incy,float *params,float *result,int blockSize) {
	transform<float>(n,idx,dy,incy,params,result,blockSize);

}
