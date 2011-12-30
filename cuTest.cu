#pragma once 

#include "cuTest.h"

extern 
void add_kernel(float* in_1, float* in_2, float* out, int arraySize, gpuConfig* config_GPU);


__global__ 
void vecAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

void add_kernel(float* in_1, float* in_2, float* out, int arraySize, gpuConfig* config_GPU)
{
	vecAdd <<< config_GPU->numBlocksPerGrid, config_GPU->numThreadsPerBlock >>> (in_1, in_2, out, arraySize);
}
