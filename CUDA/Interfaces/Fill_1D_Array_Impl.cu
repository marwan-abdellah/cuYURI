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

#ifndef _FILL_1D_ARRAY_IMPL_CU_
#define _FILL_1D_ARRAY_IMPL_CU_

#include "cuGlobals.h"
#include "Timers/Boost.h"

#include "Kernels/Fill_1D_Array_Kernel.cu"

template <typename T>
extern
void cu_Fill_1D_Array_Impl
(dim3 cuBlock, dim3 cuGrid, T *devArray, int N,
 int fillOrder, cuProfile* profile)
{
    // Create CUDA timer
    cutCreateTimer(&(profile->kernelTime));

    // Reset CUDA timer
    cutResetTimer(profile->kernelTime);

    // Start CUDA timer
    cutStartTimer(profile->kernelTime);

    // Execute the kernel
    Fill_1D_Array_Kernel <<< cuGrid, cuBlock >>> (devArray, N, fillOrder);

    // Stop CUDA timer
    cutStopTimer(profile->kernelTime);

    // Calculate kernel execution time
    profile->kernelDuration = cutGetTimerValue(profile->kernelTime);

    // Check successfull execution of the kernel
    profile->kernelExecErr = cudaPeekAtLastError();
}

template
void cu_Fill_1D_Array_Impl <char>
(dim3 cuBlock, dim3 cuGrid,
char *devArray, int N,
int fillOrder, cuProfile* profile);

template
void cu_Fill_1D_Array_Impl <unsigned char>
(dim3 cuBlock, dim3 cuGrid,
unsigned char *devArray, int N,
int fillOrder, cuProfile* profile);

template
void cu_Fill_1D_Array_Impl <int>
(dim3 cuBlock, dim3 cuGrid,
int *devArray, int N,
int fillOrder, cuProfile* profile);

template
void cu_Fill_1D_Array_Impl <unsigned int>
(dim3 cuBlock, dim3 cuGrid,
unsigned int *devArray, int N,
int fillOrder, cuProfile* profile);

template
void cu_Fill_1D_Array_Impl <float>
(dim3 cuBlock, dim3 cuGrid,
float *devArray, int N,
int fillOrder, cuProfile* profile);

template
void cu_Fill_1D_Array_Impl <double>
(dim3 cuBlock, dim3 cuGrid,
double *devArray, int N,
int fillOrder, cuProfile* profile);

#endif // _FILL_1D_ARRAY_IMPL_CU_
