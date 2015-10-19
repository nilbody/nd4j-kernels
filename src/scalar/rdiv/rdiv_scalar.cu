#include <scalar.h>
//scalar and current element
template<> __device__ double op<double>(double d1,double d2,double *params) {
   return d1 / d2;
}

extern "C"
__global__ void rdiv_scalar_double(int n, int idx,double dx,double *dy,int incy,double *params,double *result,int blockSize) {
       transform<double>(n,idx,dx,dy,incy,params,result,blockSize);
 }


template<> __device__ float op<float>(float d1,float d2,float *params) {
   return d1 / d2;
}

extern "C"
__global__ void rdiv_scalar_float(int n, int idx,float dx,float *dy,int incy,float *params,float *result,int blockSize) {
       transform<float>(n,idx,dx,dy,incy,params,result,blockSize);
 }
