#include "cuArray_1D.h"

namespace cuArray_1D
{
	gridConf* configGPU; 
}
 
int* CUDA::arraySpace::array_1D::allocateArray_DEVICE(int numItems)
{
	int* arr_1D_DEV; 
	size_t arraySize; 

	arraySize = (sizeof(int*) * numItems);
	cutilSafeCall(cudaMalloc((void**)&arr_1D_DEV, arraySize));
	return arr_1D_DEV; 
}

int* CUDA::arraySpace::array_1D::fillArray_HOST_SEQ(int numItems)
{
	int* arr_1D_HOST; 
	int arraySize; 
	arraySize = (sizeof(int*) * numItems);
	arr_1D_HOST = (int*) malloc (arraySize);

	for (int i = 0; i < numItems; i++)
		arr_1D_HOST[i] = i; 

	return arr_1D_HOST; 
}

int* CUDA::arraySpace::array_1D::zeroArray_HOST(int numItems)
{
	int* arr_1D_HOST; 
	int arraySize; 
	arraySize = (sizeof(int*) * numItems);
	arr_1D_HOST = (int*) malloc (arraySize);

	for (int i = 0; i < numItems; i++)
		arr_1D_HOST[i] = 1; 

	return arr_1D_HOST; 
}

void CUDA::arraySpace::array_1D::shoveArray_HOST_TO_DEVICE(int* hostArray_1D, int* deviceArray_1D, int numItem)
{
	size_t arraySize = (sizeof (int*) * numItem); 
	cutilSafeCall(cudaMemcpy(deviceArray_1D, hostArray_1D, arraySize, cudaMemcpyHostToDevice));
}

void CUDA::arraySpace::array_1D::shoveArray_DEVICE_HOST(int* hostArray_1D, int* deviceArray_1D, int numItem)
{
	size_t arraySize = (sizeof (int*) * numItem); 
	cutilSafeCall(cudaMemcpy(hostArray_1D, deviceArray_1D, arraySize, cudaMemcpyDeviceToHost));
}

void CUDA::arraySpace::array_1D::setGPU_Config(int numThreads_Total, int numThreads_PerBlock, bool POWER_OF_TWO_THREAD_NUM)
{
	cuArray_1D::configGPU = CUDA::GPU_CONFIG::config_1D(numThreads_Total, numThreads_PerBlock, POWER_OF_TWO_THREAD_NUM); 
}

void CUDA::arraySpace::array_1D::runKernel_Add_1D(const int* inArr_1, const int* inArr_2, int* outArr, int numItem)
{
	cuAddArray_1D(inArr_1, inArr_2, outArr, numItem, cuArray_1D::configGPU);
}

