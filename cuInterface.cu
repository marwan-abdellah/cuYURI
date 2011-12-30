#include "cuShared.h"

#include "cuKernel.cu"

extern 
void add_2D_Array(float* inArr_1, float* inArr_2, float* outArr, int arrayWidth, int arrayHeight, pitchSize __pitch, dim3 __numBlocks, dim3 __numThreadsPerBlock)
{
	add_2D_Flat <<< __numBlocks, __numThreadsPerBlock >>>(inArr_1, inArr_2, outArr, __pitch, arrayWidth, arrayHeight);

}