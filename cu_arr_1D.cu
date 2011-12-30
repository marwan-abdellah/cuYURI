/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: cu_arr_1D.cu
	Description		:
	Created			: 
	Note(s)			:
***************************************************************************
*/


#pragma once 

#include "cuShared.h"
#include "cu_arr_1D_add.cu"
#include "cu_arr_1D_mul.cu"
#include "cu_arr_1D_sub.cu"
#include "cu_arr_1D_div.cu"

// ***************************************************************************
extern "C"
void add_i_1D(int* in_1, int* in_2, int* out, int arraySize, gpuConfig_1D* conf)
{
	addArrays_kernel__i <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, arraySize);
}

extern "C"
void add_f_1D(float* in_1, float* in_2, float* out, int arraySize, gpuConfig_1D* conf)
{
	addArrays_kernel__f <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, arraySize);
}

extern "C"
void add_d_1D(double* in_1, double* in_2, double* out, int arraySize, gpuConfig_1D* conf)
{
	addArrays_kernel__d <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, arraySize);
}

// ***************************************************************************
extern "C"
void mul_i_1D(int* in_1, int* in_2, int* out, int arraySize, gpuConfig_1D* conf)
{
	mulArrays_kernel__i <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, arraySize);
}

extern "C"
void mul_f_1D(float* in_1, float* in_2, float* out, int arraySize, gpuConfig_1D* conf)
{
	mulArrays_kernel__f <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, arraySize);
}

extern "C"
void mul_d_1D(double* in_1, double* in_2, double* out, int arraySize, gpuConfig_1D* conf)
{
	mulArrays_kernel__d <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, arraySize);
}

// ***************************************************************************
extern "C"
void sub_i_1D(int* in_1, int* in_2, int* out, int arraySize, gpuConfig_1D* conf)
{
	subArrays_kernel__i <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, arraySize);
}

extern "C"
void sub_f_1D(float* in_1, float* in_2, float* out, int arraySize, gpuConfig_1D* conf)
{
	subArrays_kernel__f <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, arraySize);
}

extern "C"
void sub_d_1D(double* in_1, double* in_2, double* out, int arraySize, gpuConfig_1D* conf)
{
	subArrays_kernel__d <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, arraySize);
}

// ***************************************************************************
extern "C"
void div_i_1D(int* in_1, int* in_2, int* out, int arraySize, gpuConfig_1D* conf)
{
	divArrays_kernel__i <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, arraySize);
}

extern "C"
void div_f_1D(float* in_1, float* in_2, float* out, int arraySize, gpuConfig_1D* conf)
{
	divArrays_kernel__f <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, arraySize);
}

extern "C"
void div_d_1D(double* in_1, double* in_2, double* out, int arraySize, gpuConfig_1D* conf)
{
	divArrays_kernel__d <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, arraySize);
}
