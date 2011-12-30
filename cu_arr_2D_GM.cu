/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: cu_arr_2D_GM.cu
	Description		:
	Created			: 
	Note(s)			:
***************************************************************************
*/

#pragma once 

#include "cuShared.h"
#include "cu_arr_2D_add_GM.cu"
#include "cu_arr_2D_mul_GM.cu"
#include "cu_arr_2D_sub_GM.cu"
#include "cu_arr_2D_div_GM.cu"

// ***************************************************************************
extern "C"
void add_i_2D_GM(int* in_1, int* in_2, int* out, int arrayWidth, int arrayHeight, size_t __pitch, gpuConfig_XD* conf)
{
	addArrays_kernel__i <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, __pitch, arrayWidth, arrayHeight);
}

extern "C"
void add_f_2D_GM(float* in_1, float* in_2, float* out, int arrayWidth, int arrayHeight, size_t __pitch, gpuConfig_XD* conf)
{
	addArrays_kernel__f <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, __pitch, arrayWidth, arrayHeight);
}

extern "C"
void add_d_2D_GM(double* in_1, double* in_2, double* out, int arrayWidth, int arrayHeight, size_t __pitch, gpuConfig_XD* conf)
{
	addArrays_kernel__d <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, __pitch, arrayWidth, arrayHeight);
}

// ***************************************************************************
extern "C"
void mul_i_2D_GM(int* in_1, int* in_2, int* out, int arrayWidth, int arrayHeight, size_t __pitch, gpuConfig_XD* conf)
{
	mulArrays_kernel__i <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, __pitch, arrayWidth, arrayHeight);
}

extern "C"
void mul_f_2D_GM(float* in_1, float* in_2, float* out, int arrayWidth, int arrayHeight, size_t __pitch, gpuConfig_XD* conf)
{
	mulArrays_kernel__f <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, __pitch, arrayWidth, arrayHeight);
}

extern "C"
void mul_d_2D_GM(double* in_1, double* in_2, double* out, int arrayWidth, int arrayHeight, size_t __pitch, gpuConfig_XD* conf)
{
	mulArrays_kernel__d <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, __pitch, arrayWidth, arrayHeight);
}

// ***************************************************************************
extern "C"
void sub_i_2D_GM(int* in_1, int* in_2, int* out, int arrayWidth, int arrayHeight, size_t __pitch, gpuConfig_XD* conf)
{
	subArrays_kernel__i <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, __pitch, arrayWidth, arrayHeight);
}

extern "C"
void sub_f_2D_GM(float* in_1, float* in_2, float* out, int arrayWidth, int arrayHeight, size_t __pitch, gpuConfig_XD* conf)
{
	subArrays_kernel__f <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, __pitch, arrayWidth, arrayHeight);
}

extern "C"
void sub_d_2D_GM(double* in_1, double* in_2, double* out, int arrayWidth, int arrayHeight, size_t __pitch, gpuConfig_XD* conf)
{
	subArrays_kernel__d <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, __pitch, arrayWidth, arrayHeight);
}

// ***************************************************************************
extern "C"
void div_i_2D_GM(int* in_1, int* in_2, int* out, int arrayWidth, int arrayHeight, size_t __pitch, gpuConfig_XD* conf)
{
	divArrays_kernel__i <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, __pitch, arrayWidth, arrayHeight);
}

extern "C"
void div_f_2D_GM(float* in_1, float* in_2, float* out, int arrayWidth, int arrayHeight, size_t __pitch, gpuConfig_XD* conf)
{
	divArrays_kernel__f <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, __pitch, arrayWidth, arrayHeight);
}

extern "C"
void div_d_2D_GM(double* in_1, double* in_2, double* out, int arrayWidth, int arrayHeight, size_t __pitch, gpuConfig_XD* conf)
{
	divArrays_kernel__d <<< conf->numBlocksPerGrid, conf->numThreadsPerBlock >>> (in_1, in_2, out, __pitch, arrayWidth, arrayHeight);
}
