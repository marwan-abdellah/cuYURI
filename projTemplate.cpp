// Includes
#include <stdio.h>
#include <cutil_inline.h>
#include <shrQATest.h>

extern void cuFun(const float* A, const float* B, float* C, int arraySize); 

namespace Host
{
	float* createArray(int arrSize)
	{
		float* ptrArray = (float*) malloc(sizeof(float*) * arrSize); 
		return ptrArray; 
	}

	void freeArray(float* ptrArray)
	{
		if (ptrArray)
			free(ptrArray); 
	}
}

namespace Device
{
	void initDevice(int argc, char** argv)
	{
		shrQAStart(argc, argv);
	}
	
	float* createArray(int arrSize)
	{
		float* ptrArray; 
		size_t dataSize = (sizeof(float*) * arrSize); 
		cutilSafeCall(cudaMalloc((void**)&ptrArray, dataSize)); 
		return ptrArray; 
	}
	
	void freeArray(float* ptrArray)
	{
		if (ptrArray)
			cudaFree(ptrArray); 
	}
}

namespace Util
{
	/*
	void verifyRes(float* ptrDevArr_Host, float* ptrHostArr, int arrSize)
	{
		float sum = inVec1_H[i] + inVec2_H[i];
		for (int i = 0; i < arrSize; i++)
		{
			if (ptrDevArr_Host[i] != ptrHostArr[i])
				prinf("Error"); 
		}		
	}*/
	
	void printArray(float* ptrArray, int arrSize)
	{
		for (int i = 0; i < arrSize; i++)
			printf("@[%d] %d \n", i, ptrArray[i]); 
	}
	
	

}

namespace Memory
{
	void uploadArray(float* ptrArray_D, float* ptrArray_H, int arrSize)
	{
		 size_t dataSize = (sizeof(float) * arrSize);
		 cutilSafeCall(cudaMemcpy(ptrArray_D, ptrArray_H, dataSize, cudaMemcpyHostToDevice));
	}
	
	void downloadArray(float* ptrArray_D, float* ptrArray_H, int arrSize)
	{
		 size_t dataSize = (sizeof(float) * arrSize); 
		 cutilSafeCall(cudaMemcpy(ptrArray_H, ptrArray_D, dataSize, cudaMemcpyDeviceToHost));
	}
}








namespace exAdd
{
	int arraySize; 
	size_t dataSize;
	
	float* inVec1_H;
	float* inVec2_H;
	float* outVec_H;
	float* inVec1_D;
	float* inVec2_D;
	float* outVec_D;
	
	int numThreadsPerBlock;
	int numBlocksPerGrid;  
} 



bool noprompt = false;

// Functions
void CleanupResources(void);
void RandomInit(float*, int);
void ParseArguments(int, char**);




// Host code
int main(int argc, char** argv)
{
    Device::initDevice(argc, argv); 
    
    exAdd::arraySize = 1024;
    exAdd::dataSize = exAdd::arraySize * sizeof(float);
    
    exAdd::inVec1_H = Host::createArray(exAdd::arraySize);
    exAdd::inVec2_H = Host::createArray(exAdd::arraySize);
    exAdd::outVec_H = Host::createArray(exAdd::arraySize);
   
    RandomInit(exAdd::inVec1_H, exAdd::arraySize);
    RandomInit(exAdd::inVec2_H, exAdd::arraySize);
 
    exAdd::inVec1_H = Device::createArray(exAdd::arraySize);
    exAdd::inVec2_H = Device::createArray(exAdd::arraySize);
    exAdd::outVec_H = Device::createArray(exAdd::arraySize);

    Memory::uploadArray(exAdd::inVec1_D, exAdd::inVec1_H, exAdd::arraySize);
    Memory::uploadArray(exAdd::inVec2_D, exAdd::inVec2_H, exAdd::arraySize); 

    exAdd::numThreadsPerBlock = 256;
    exAdd::numBlocksPerGrid = (exAdd::arraySize + exAdd::numThreadsPerBlock - 1) / exAdd::numThreadsPerBlock;
    
    vecAdd <<< exAdd::numBlocksPerGrid, exAdd::numThreadsPerBlock >>>(exAdd::inVec1_D, exAdd::inVec2_D, exAdd::outVec_D, exAdd::arraySize);
    cutilCheckMsg("kernel launch failure");

    cutilSafeCall(cutilDeviceSynchronize());

    Memory::downloadArray(exAdd::outVec_H, exAdd::outVec_D, exAdd::arraySize);
    
    Util::printArray(exAdd::outVec_H, exAdd::arraySize); 
    
	Host::freeArray(exAdd::inVec1_H); 
	Host::freeArray(exAdd::inVec2_H); 
	Host::freeArray(exAdd::outVec_H); 
	
	Device::freeArray(exAdd::inVec1_D);
	Device::freeArray(exAdd::inVec2_D);
	Device::freeArray(exAdd::outVec_D);
	
	cutilDeviceReset();
}


// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

