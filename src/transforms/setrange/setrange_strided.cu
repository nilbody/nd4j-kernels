#include <transform.h>


__device__ double op(double d1,double *params) {
	double min = params[0];
	double max = params[1];
	if(d1 >= min && d1 <= max)
		return d1;
	if(min == 0 && max == 1) {
		double val = 1 / (1 + exp(-d1));
		return (floor(val * (max - min)) + min);
	}

	double ret =  (floor(d1 * (max - min)) + min);
	return ret;

}

extern "C"
__global__ void setrange_strided_double(int n,int idx,double *dy,int incy,double *params,double *result,int blockSize) {
	transform<double>(n,idx,dy,incy,params,result,blockSize);

}

__device__ float op(float d1,float *params) {
	float min = params[0];
	float max = params[1];
	if(d1 >= min && d1 <= max)
		return d1;
	if(min == 0 && max == 1) {
		float val = 1 / (1 + expf(-d1));
		return (floorf(val * (max - min)) + min);
	}

	float ret =  (floorf(d1 * (max - min)) + min);
	return ret;

}

extern "C"
__global__ void setrange_strided_float(int n,int idx,float *dy,int incy,float *params,float *result,int blockSize) {
	transform<float>(n,idx,dy,incy,params,result,blockSize);

}
