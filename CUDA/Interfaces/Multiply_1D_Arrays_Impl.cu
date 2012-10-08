/*********************************************************************
 * Copyright © 2011-2012,
 * Marwan Abdellah: <abdellah.marwan@gmail.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation.

 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.

 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 ********************************************************************/


#ifndef _MULTIPLY_1D_ARRAY_DEVICE_IMPL_CU_
#define _MULTIPLY_1D_ARRAY_DEVICE_IMPL_CU_

#include "cuGlobals.h"
#include "Timers/Boost.h"

#include "Kernels/Multiply_1D_Arrays_Kernel.cu"

template <typename T>
extern
void cu_Multiply_1D_Arrays_Impl
(dim3 cuBlock, dim3 cuGrid,
 T *devArrayInput_1, T* devArrayInput_2, T* devArrayOutput, int N,
 cuProfile* profile)
{
    // Create CUDA timer
    cutCreateTimer(&(profile->kernelTime));

    // Reset CUDA timer
    cutResetTimer(profile->kernelTime);

    // Start CUDA timer
    cutStartTimer(profile->kernelTime);

    // Execute the kernel
    Multiply_1D_Arrays_Kernel
            <<< cuGrid, cuBlock >>> (devArrayInput_1, devArrayInput_2, devArrayOutput, N);

    // Stop CUDA timer
    cutStopTimer(profile->kernelTime);

    // Calculate kernel execution time
    profile->kernelDuration = cutGetTimerValue(profile->kernelTime);

    // Check successfull execution of the kernel
    profile->kernelExecErr = cudaPeekAtLastError();
}

template
void cu_Multiply_1D_Arrays_Impl <char>
(dim3 cuBlock, dim3 cuGrid,
char *devArrayInput_1, char *devArrayInput_2, char* devArrayOutput, int N,
cuProfile* profile);

template
void cu_Multiply_1D_Arrays_Impl <unsigned char>
(dim3 cuBlock, dim3 cuGrid,
unsigned char *devArrayInput_1, unsigned char *devArrayInput_2, unsigned char* devArrayOutput, int N,
cuProfile* profile);

template
void cu_Multiply_1D_Arrays_Impl <int>
(dim3 cuBlock, dim3 cuGrid,
int *devArrayInput_1, int *devArrayInput_2, int* devArrayOutput, int N,
cuProfile* profile);

template
void cu_Multiply_1D_Arrays_Impl <unsigned int>
(dim3 cuBlock, dim3 cuGrid,
unsigned int *devArrayInput_1, unsigned int *devArrayInput_2, unsigned int* devArrayOutput, int N,
cuProfile* profile);

template
void cu_Multiply_1D_Arrays_Impl <float>
(dim3 cuBlock, dim3 cuGrid,
float *devArrayInput_1, float *devArrayInput_2, float* devArrayOutput, int N,
cuProfile* profile);

template
void cu_Multiply_1D_Arrays_Impl <double>
(dim3 cuBlock, dim3 cuGrid,
double *devArrayInput_1, double *devArrayInput_2, double* devArrayOutput, int N,
cuProfile* profile);

#endif // _MULTIPLY_1D_ARRAY_DEVICE_IMPL_CU_
