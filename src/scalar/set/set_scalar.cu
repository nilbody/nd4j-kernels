#include <scalar.h>

template<> __device__ double op<double>(double d1,double d2,double *params) {
   return d2;
}


__global__ void set_scalar_double(int n, int idx,double dx,double *dy,int incy,double *params,double *result,int blockSize) {
       transform<double>(n,idx,dx,dy,incy,params,result,blockSize);
 }


template<> __device__ float op<float>(float d1,float d2,float *params) {
   return d2;
}


__global__ void set_scalar_float(int n, int idx,float dx,float *dy,int incy,float *params,float *result,int blockSize) {
       transform<float>(n,idx,dx,dy,incy,params,result,blockSize);
 }
