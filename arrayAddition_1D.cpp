#include "arrayAddition_1D.h"

void printArray_1D(int* arr_1D, int numItems)
{
	for (int i = 0; i < numItems; i++)
		printf("@[%d] = %d \n", i, arr_1D[i]); 
}

/**************
 * Parameters 
 **************/
namespace arrayAddition_1D
{
	int* inArr_1_1D_CPU; 
	int* inArr_2_1D_CPU;
	int* outArr_1D_CPU;

	int* inArr_1_1D_GPU_H; 
	int* inArr_2_1D_GPU_H;
	int* outArr_1D_GPU_H;

	int* inArr_1_1D_GPU_D; 
	int* inArr_2_1D_GPU_D;
	int* outArr_1D_GPU_D;

	int numElements;
	int numThreadsPerBlock; 

	bool NUM_SEQ; /* 0 = RND, 1 = SEQ */
	bool VALID_FLAG = true;
	bool POWER_OF_TWO_THREAD_NUM; 
}

void Examples::Array_1D::Addition::initApp(int numElements, int numThreadsPerBlock, bool NUM_SEQ)
{
	arrayAddition_1D::numElements = numElements;
	arrayAddition_1D::numThreadsPerBlock = numThreadsPerBlock; 

	if (numThreadsPerBlock / 2)
		arrayAddition_1D::POWER_OF_TWO_THREAD_NUM = true; 

	arrayAddition_1D::NUM_SEQ = NUM_SEQ; 
}

void Examples::Array_1D::Addition::init_CUDA_Context(int argc, char** argv)
{
	shrQAStart(argc, argv); 
}

void Examples::Array_1D::Addition::runCPU_Kernel()
{ 	
	arrayAddition_1D::inArr_1_1D_CPU = arraySpace::array_1D::createArray(arrayAddition_1D::numElements); 
	arrayAddition_1D::inArr_2_1D_CPU = arraySpace::array_1D::createArray(arrayAddition_1D::numElements); 
	arrayAddition_1D::outArr_1D_CPU  = arraySpace::array_1D::createArray(arrayAddition_1D::numElements); 

	if (arrayAddition_1D::NUM_SEQ)
	{
		arraySpace::array_1D::fillArray_SEQ(arrayAddition_1D::inArr_1_1D_CPU, arrayAddition_1D::numElements);
		arraySpace::array_1D::fillArray_SEQ(arrayAddition_1D::inArr_2_1D_CPU, arrayAddition_1D::numElements);
	}
	else 
	{
		arraySpace::array_1D::fillArray_RND(arrayAddition_1D::inArr_1_1D_CPU, arrayAddition_1D::numElements);
		arraySpace::array_1D::fillArray_RND(arrayAddition_1D::inArr_2_1D_CPU, arrayAddition_1D::numElements);
	}

	arraySpace::array_1D::zeroArray(arrayAddition_1D::outArr_1D_CPU, arrayAddition_1D::numElements);

	arraySpace::array_1D::addArrays(arrayAddition_1D::inArr_1_1D_CPU, arrayAddition_1D::inArr_2_1D_CPU, arrayAddition_1D::outArr_1D_CPU, arrayAddition_1D::numElements);		

	printArray_1D(arrayAddition_1D::inArr_1_1D_CPU, arrayAddition_1D::numElements); 
	printArray_1D(arrayAddition_1D::inArr_2_1D_CPU, arrayAddition_1D::numElements); 
	printArray_1D(arrayAddition_1D::outArr_1D_CPU, arrayAddition_1D::numElements); 

}

void Examples::Array_1D::Addition::runGPU_Kernel()
{
	
	arrayAddition_1D::inArr_1_1D_GPU_D = CUDA::arraySpace::array_1D::allocateArray_DEVICE(arrayAddition_1D::numElements); 
	arrayAddition_1D::inArr_2_1D_GPU_D = CUDA::arraySpace::array_1D::allocateArray_DEVICE(arrayAddition_1D::numElements); 
	arrayAddition_1D::outArr_1D_GPU_D = CUDA::arraySpace::array_1D::allocateArray_DEVICE(arrayAddition_1D::numElements);

	CUDA::arraySpace::array_1D::setGPU_Config(arrayAddition_1D::numElements, arrayAddition_1D::numThreadsPerBlock, arrayAddition_1D::POWER_OF_TWO_THREAD_NUM);

	arrayAddition_1D::inArr_1_1D_GPU_H	= CUDA::arraySpace::array_1D::fillArray_HOST_SEQ(arrayAddition_1D::numElements); 
	arrayAddition_1D::inArr_2_1D_GPU_H	= CUDA::arraySpace::array_1D::fillArray_HOST_SEQ(arrayAddition_1D::numElements);
	arrayAddition_1D::outArr_1D_GPU_H	= CUDA::arraySpace::array_1D::zeroArray_HOST(arrayAddition_1D::numElements); 

	CUDA::arraySpace::array_1D::shoveArray_HOST_TO_DEVICE(arrayAddition_1D::inArr_1_1D_GPU_H, arrayAddition_1D::inArr_1_1D_GPU_D, arrayAddition_1D::numElements); 
	CUDA::arraySpace::array_1D::shoveArray_HOST_TO_DEVICE(arrayAddition_1D::inArr_2_1D_GPU_H, arrayAddition_1D::inArr_2_1D_GPU_D, arrayAddition_1D::numElements); 
	CUDA::arraySpace::array_1D::shoveArray_HOST_TO_DEVICE(arrayAddition_1D::outArr_1D_GPU_H, arrayAddition_1D::outArr_1D_GPU_D, arrayAddition_1D::numElements); 

	CUDA::arraySpace::array_1D::runKernel_Add_1D(arrayAddition_1D::inArr_1_1D_GPU_D,arrayAddition_1D::inArr_2_1D_GPU_D, arrayAddition_1D::outArr_1D_GPU_D, arrayAddition_1D::numElements);  

	CUDA::arraySpace::array_1D::shoveArray_DEVICE_HOST(arrayAddition_1D::outArr_1D_GPU_H, arrayAddition_1D::outArr_1D_GPU_D, arrayAddition_1D::numElements); 
}

void Examples::Array_1D::Addition::validateResults()
{
	 for (int i = 0; i < arrayAddition_1D::numElements; i++)
	 {
		 printf("@%d, CPU %d, GPU %d \n", i, arrayAddition_1D::outArr_1D_CPU[i], arrayAddition_1D::outArr_1D_GPU_H[i]); 

		 if (arrayAddition_1D::outArr_1D_CPU[i] - arrayAddition_1D::outArr_1D_GPU_H[i] != 0)
		 {
			 printf("Results are not matching @ index [%d] \n", i);
			 arrayAddition_1D::VALID_FLAG = false;  
		 }
	 }

	 if (arrayAddition_1D::VALID_FLAG)
		printf("TEST PASSED SUCCESSFULLY ... \n"); 
	 else 
		printf("TEST FAILED ... \n");

}

void Examples::Array_1D::Addition::sysExit()
{
	arraySpace::array_1D::freeArray(arrayAddition_1D::inArr_1_1D_CPU); 
	arraySpace::array_1D::freeArray(arrayAddition_1D::inArr_2_1D_CPU); 
	arraySpace::array_1D::freeArray(arrayAddition_1D::outArr_1D_CPU); 
	arraySpace::array_1D::freeArray(arrayAddition_1D::inArr_1_1D_GPU_H); 
	arraySpace::array_1D::freeArray(arrayAddition_1D::inArr_2_1D_GPU_H); 
	arraySpace::array_1D::freeArray(arrayAddition_1D::outArr_1D_GPU_H); 

	// Exit execution contexts.
	cutilDeviceReset();
}

void Examples::Array_1D::Addition::runExample(int numElements, int numThreadsPerBlock, bool NUM_SEQ, int argc, char** argv)
{
	Examples::Array_1D::Addition::initApp(numElements, numThreadsPerBlock, NUM_SEQ); 
	Examples::Array_1D::Addition::init_CUDA_Context(argc, argv); 
	Examples::Array_1D::Addition::runCPU_Kernel();
	Examples::Array_1D::Addition::runGPU_Kernel(); 
	Examples::Array_1D::Addition::validateResults();
	Examples::Array_1D::Addition::sysExit(); 	
}


