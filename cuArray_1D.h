#pragma once 

#include "shared.h"
#include "configGPU.h"

extern 
void cuAddArray_1D(const int* inArr_1, const int* inArr_2, int* outArr, int numItem, gridConf* configGPU);  

namespace CUDA
{
	namespace arraySpace
	{
		namespace array_1D
		{
			int* allocateArray_DEVICE(int numItems);
			int* fillArray_HOST_SEQ(int numItems); 
			int* zeroArray_HOST(int numItems); 
			void shoveArray_HOST_TO_DEVICE(int* hostArray_1D, int* deviceArray_1D, int numItem);
			void shoveArray_DEVICE_HOST(int* hostArray_1D, int* deviceArray_1D, int numItem);
			void setGPU_Config(int numThreads_Total, int numThreads_PerBlock, bool POWER_OF_TWO_THREAD_NUM); 
			void runKernel_Add_1D(const int* inArr_1, const int* inArr_2, int* outArr, int numItem);

			// void destroyArray(int* arr_1D); 
		}
	}
}
 