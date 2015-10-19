#include <pairwise_transform.h>

#define MIN 1e-12


template <> __device__ double op<double>(double d1,double d2,double *params) {
	double diff = d1 - d2;
	double absDiff = abs(diff);
	if(absDiff < MIN)
		return 1;
	return 0;
}
template <>  __device__ double op<double>(double d1,double *params) {
	return d1;
}


__global__ void eps_strided_double(int n, int xOffset,int yOffset,double *dx, double *dy,int incx,int incy,double *params,double *result,int incz,int blockSize) {
	transform<double>(n,xOffset,yOffset,dx,dy,incx,incy,params,result,incz,blockSize);

}





template <>  __device__ float op<float>(float d1,float d2,float *params) {
	float diff = d1 - d2;
	float absDiff = fabsf(diff);
	if(absDiff < MIN)
		return 1;
	return 0;
}
template <>  __device__ float op<float>(float d1,float *params) {
	return d1;
}


__global__ void eps_strided_float(int n, int xOffset,int yOffset,float *dx, float *dy,int incx,int incy,float *params,float *result,int incz,int blockSize) {
	transform<float>(n,xOffset,yOffset,dx,dy,incx,incy,params,result,incz,blockSize);

}
