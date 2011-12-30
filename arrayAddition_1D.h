#pragma once

#include "shared.h"
#include "array_1D.h"
#include "cuArray_1D.h"

using namespace arraySpace; 

namespace Examples
{
	namespace Array_1D
	{
		namespace Addition
		{
			void initApp(int numElements, int numThreadsPerBlock, bool NUM_SEQ); 
			void init_CUDA_Context(int argc, char** argv); 
			void runCPU_Kernel(); 
			void runGPU_Kernel(); 
			void validateResults();
			void sysExit(); 
			
			void runExample(int numElements, int numThreadsPerBlock, bool NUM_SEQ, int argc, char** argv); 
		}
	}
}




