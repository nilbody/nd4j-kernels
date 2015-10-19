#include <scalar.h>
//scalar and current element
template<> __device__ double op<double>(double d1,double d2,double *params) {
    if(d2 >= d1) {return 1;}
    return 0;

}


__global__ void greaterthanorequal_scalar_double(int n, int idx,double dx,double *dy,int incx,double *params,double *result,int blockSize) {
       transform<double>(n,idx,dx,dy,incx,params,result,blockSize);
}


template<> __device__ float op<float>(float d1,float d2,float *params) {
    if(d2 >= d1) {return 1;}
    return 0;

}


__global__ void greaterthanorequal_scalar_float(int n, int idx,float dx,float *dy,int incx,float *params,float *result,int blockSize) {
       transform<float>(n,idx,dx,dy,incx,params,result,blockSize);
}


