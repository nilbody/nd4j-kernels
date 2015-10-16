#include <reduce.h>

#include "helper_cuda.h"

template <>   __device__ double merge<double>(double old,double opOutput,double *extraParams) {
	return max(old,opOutput);
}


template <> __device__ double update<double>(double old,double opOutput,double *extraParams) {
	return max(old,opOutput);
}


template <> __device__ double op<double>(double d1,double *extraParams) {
	return d1;
}


template <>  __device__ double postProcess<double>(double reduction,int n,int xOffset,double *dx,int incx,double *extraParams,double *result) {
	return reduction;
}

template <>   __device__ float merge<float>(float old,float opOutput,float *extraParams) {
	return max(old,opOutput);
}


template <> __device__ float update<float>(float old,float opOutput,float *extraParams) {
	return max(old,opOutput);
}


template <> __device__ float op<float>(float d1,float *extraParams) {
	return d1;
}


template <>  __device__ float postProcess<float>(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
	return reduction;
}

__global__ void max_strided_double(
		int n
		,double *dx
		,int *xVectorInfo
		,double *extraParams
		,double *result,
		int *resultVectorInfo
		,int *gpuInformation,
		int *problemDefinition) {
	transform<double>(n,dx,xVectorInfo,extraParams,result,resultVectorInfo,gpuInformation,problemDefinition);
}




__device__ float merge(float old,float opOutput,float *extraParams) {
	return fmaxf(old,opOutput);
}


__device__ float update(float old,float opOutput,float *extraParams) {
	return fmaxf(old,opOutput);
}


__device__ float op(float d1,float *extraParams) {
	return d1;
}


__device__ float postProcess(float reduction,int n,int xOffset,float *dx,int incx,float *extraParams,float *result) {
	return reduction;
}


__global__ void max_strided_float(
		int n
		,float *dx
		,int *xVectorInfo
		,float *extraParams
		,float *result,
		int *resultVectorInfo
		,int *gpuInformation,
		int *problemDefinition) {
	transform<float>(n,dx,xVectorInfo,extraParams,result,resultVectorInfo,gpuInformation,problemDefinition);
}





void setupArrayInfo(int **arrInfo,int **gpuArrInfo,int offset, int incx,int xLength,int xElementWiseStride) {
	int dataLength = 4 * sizeof(int);
	*arrInfo = (int *) malloc(dataLength);
	int *otherPointer = *arrInfo;
	otherPointer[0] = offset;
	otherPointer[1] = incx;
	otherPointer[2] = xLength;
	otherPointer[3] = xElementWiseStride;
	checkCudaErrors(cudaMalloc((void **) gpuArrInfo,dataLength));
	for(int i = 0; i < 4; i++) {
		printf("Value for shape %d\n",otherPointer[i]);
	}

	int *gpuDeReferenced = *gpuArrInfo;

	printf("Copied shape data to gpu\n");
	checkCudaErrors(cudaMemcpy(gpuDeReferenced,otherPointer,dataLength,cudaMemcpyHostToDevice));
	printf("About to sync on cuda memcopy\n");
	checkCudaErrors(cudaDeviceSynchronize());

	printf("Trying to copy shape data from gpu\n");
	int *testData = (int *) malloc(dataLength);
	checkCudaErrors(cudaMemcpy(otherPointer,gpuDeReferenced,dataLength,cudaMemcpyDeviceToHost));


}


void setupData(int length ,double **data,double **gdata) {
	*data = (double *) calloc(length,sizeof(double));
	double *derefedData = *data;
	for(int i = 0; i < length; i++) {
		derefedData[i] = i + 1;
	}

	checkCudaErrors(cudaMalloc((void **) gdata,length * sizeof(double)));

	double *dereffedGpu = *gdata;

	checkCudaErrors(cudaMemcpy(dereffedGpu,derefedData,length * sizeof(double),cudaMemcpyHostToDevice));


}


void setupParams(int paramsLength,double **params,double **gParams) {
	*params = (double *) calloc(paramsLength,sizeof(double));
	double *dereffedParams = *params;
	dereffedParams[1] = 5.4772257804870605;
	dereffedParams[2] = 5.4772257804870605;

	checkCudaErrors(cudaMalloc((void **) gParams,paramsLength * sizeof(double)));
	double *deReffedGpu = *gParams;

	checkCudaErrors(cudaMemcpy(deReffedGpu,dereffedParams,paramsLength * sizeof(double),cudaMemcpyHostToDevice));

}


void setupResult(int resultLength,double **result,double **gResult) {
	*result = (double *) calloc(resultLength,sizeof(double));
	checkCudaErrors(cudaMalloc((void **) gResult,resultLength * sizeof(double)));
	double *dereffedResult = *result;
	double *dereffedGpu = *gResult;
	checkCudaErrors(cudaMemcpy(dereffedGpu,dereffedResult,resultLength * sizeof(double),cudaMemcpyHostToDevice));
}


void setupGpuInfo(int **pointerInfo,int **gpuPointerInfo,int gridSize,int blockSize,int sMemSize,int stream) {
	*pointerInfo = (int *) calloc(4,sizeof(int));
	checkCudaErrors(cudaMalloc((void **) gpuPointerInfo,sizeof(int) * 4));
	int *dereffedPointerInfo = *pointerInfo;
	dereffedPointerInfo[0] = gridSize;
	dereffedPointerInfo[1] = blockSize;
	dereffedPointerInfo[2] = sMemSize;
	dereffedPointerInfo[3] = stream;

	int *dereffedGpu = *gpuPointerInfo;
	checkCudaErrors(cudaMemcpy(dereffedGpu,dereffedPointerInfo,4 * sizeof(int),cudaMemcpyHostToDevice));

}


void setupProblemDef(int **problemDef,int **gpuProblemDef,int elementsPerVector,int numberOfVectors) {
	*problemDef = (int *) calloc(2,sizeof(int));
	int *dereffedProblem = *problemDef;
	dereffedProblem[0] = elementsPerVector;
	dereffedProblem[1] = numberOfVectors;
	checkCudaErrors(cudaMalloc((void **) gpuProblemDef,2 * sizeof(int)));
	int *dereffeddGpu = *gpuProblemDef;
	checkCudaErrors(cudaMemcpy(dereffeddGpu,dereffedProblem,2 * sizeof(int),cudaMemcpyHostToDevice));

}

int main(int argc, char **argv) {
	int length = 1000;
	int resultLength = 2;
	int paramsLength = 3;
	int stride = 1;


	int numSMs;
	checkCudaErrors(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));


	//the shape information for the problem
	int *xInfo,*resultInfo;
	int *gpuXInfo,*gpuResultInfo;

	setupArrayInfo(&xInfo,&gpuXInfo,0,1,length,1);
	setupArrayInfo(&resultInfo,&gpuResultInfo,0,1,1,1);
	printf("Setup shape information for result and the data");



	//the data
	double *data;
	double *gdata;
	setupData(length,&data,&gdata);
	printf("Uploaded data of length %d\n",length);
	//params for execution
	double *params,*gParams;
	setupParams(paramsLength,&params,&gParams);
	printf("Setup extra parameters\n");

	//result array
	double *result;
	double *gresult;
	setupResult(resultLength,&result,&gresult);
	printf("Setup result array of result length %d\n",resultLength);
	printf("Iterating on  %d elements\n",length / stride);




	//gpu information

	int blockSize = 512;   // The launch configurator returned block size
	int minGridSize = 0; // The minimum grid size needed to achieve the
	// maximum occupancy for a full device launch
	int gridSize = length / blockSize;    // The actual grid size needed, based on input size
	cudaDeviceProp prop;

	checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
	int sharedMem = prop.sharedMemPerBlock;
	printf("Shared memory %d\n",sharedMem);
	int sMemSize = sharedMem;
	/*cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
			max_strided_double, 0,sharedMem);
	// Round up according to array size
	gridSize = (length + blockSize - 1) / blockSize;*/
	int *gpuInfo,*gGpuInfo;
	int stream = 2;
	setupGpuInfo(&gpuInfo,&gGpuInfo,gridSize,blockSize,sMemSize,stream);



	int *problemDef,*gpuProblemDef;
	int elementsPerVector = 5000;
	int numVectors = 2;
	setupProblemDef(&problemDef,&gpuProblemDef,elementsPerVector,numVectors);
	/**
	 * int n
		,double *dx
		,int *xVectorInfo
		,double *extraParams
		,double *result,
		int *resultVectorInfo
		,int *gpuInformation,
		int *problemDefinitioncheckCudaErrors
	 */


	int nkernels = 16;
	cudaStream_t *streams = (cudaStream_t *) malloc(nkernels * sizeof(cudaStream_t));
	cudaStream_t *gpuStreams;
	cudaMalloc(&gpuStreams,nkernels * sizeof(cudaStream_t));
	cudaMemcpy(gpuStreams,streams,nkernels * sizeof(cudaStream_t),cudaMemcpyDeviceToHost);
	for (int i = 0; i < nkernels; i++)	{
		checkCudaErrors(cudaStreamCreate(&(streams[i])));
	}



	max_strided_double<<<length / blockSize,blockSize,sMemSize>>>(
			length
			,gdata
			,gpuXInfo
			,gParams
			,gresult
			,gpuResultInfo
			,gGpuInfo
			,gpuProblemDef
	);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(result,gresult,resultLength * sizeof(double),cudaMemcpyDeviceToHost));

	for(int i = 0; i < resultLength; i++) {
		printf("%f\n",result[i]);
	}

	free(gpuStreams);





}

