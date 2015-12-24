/*
 * buffer.h
 *
 *  Created on: Dec 24, 2015
 *      Author: agibsonccc
 */

#ifndef BUFFER_H_
#define BUFFER_H_
template <typename T>
struct Buffer {
	int length;
	T *data;
	T *gData;
};

template <>
struct Buffer<double> {
	int length;
	double *data;
	double *gData;
};


template <>
struct Buffer<float> {
	int length;
	float *data;
	float *gData;
};

template <typename T>
__device__ size_t bufferSize(Buffer<T> *buffer) {
	int length2 = buffer->length;
	return sizeof(T) * buffer->length;
}

template <typename T>
__device__ void copyDataToGpu(Buffer<T> **buffer) {
	Buffer<T> bufferRef = *buffer;
	cudaMemCpy(bufferRef->gData,bufferRef->data,bufferSize(bufferRef),cudaMemcpyHostToDevice);
}

template <typename T>
__device__ void copyDataFromGpu(Buffer<T> **buffer) {
	Buffer<T> bufferRef = *buffer;
	cudaMemCpy(bufferRef->data,bufferRef->gData,bufferSize(bufferRef),cudaMemcpyDeviceToHost);
}

template <typename T>
__device__ void allocBuffer(Buffer<T> **buffer,int length) {
	Buffer<T> *bufferRef = *buffer;
	bufferRef->length = length;
	bufferRef->data = (T *) malloc(sizeof(T) * length);
	cudaMalloc(&bufferRef->gData,sizeof(T) * length);

}

template <typename T>
__device__ void freeBuffer(Buffer<T> **buffer) {
	Buffer<T> bufferRef = *buffer;
	delete[] bufferRef->data;
	cudaFree(bufferRef->gData);
}


#endif /* BUFFER_H_ */
