#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__
void FillArray_Kernel(T* devArray, int nX)
{

    // Thread
    int xThreadIdx = threadIdx.x;

    // Block Width
    int blockWidth = blockDim.x;

    // Thread Index 2D
    int index = blockIdx.x * blockWidth + xThreadIdx;

    devArray[index] = nX - index * 4;
}
