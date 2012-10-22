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

#ifndef _CU_ADD_1D_ARRAY_DEVICE_IMPL_CU_
#define _CU_ADD_1D_ARRAY_DEVICE_IMPL_CU_

#include "cuGlobals.h"
#include "Timers/Boost.h"

#include "Kernels/Constant_Add_1D_Array_Kernel.cu"

/*! Implementation for the Constant_Add_1D_Array_Kernel kernel.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput
 *          Input device vector.
 *
 * @param constVal
 *          Constant value to be added to the input device vector.
 *
 * @param devArrayOutput
 *          Output device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template <typename T>
extern
void cu_Constant_Add_1D_Array_Impl
(dim3 cuBlock, dim3 cuGrid,
 T* devArrayInput, T constVal, T* devArrayOutput, int N,
 cuProfile* profile)
{
    // Create CUDA timer
    cutCreateTimer(&(profile->kernelTime));

    // Reset CUDA timer
    cutResetTimer(profile->kernelTime);

    // Start CUDA timer
    cutStartTimer(profile->kernelTime);

    // Execute the kernel
    Constant_Add_1D_Array_Kernel
            <<< cuGrid, cuBlock >>> (devArrayInput, constVal, devArrayOutput, N);

    // Stop CUDA timer
    cutStopTimer(profile->kernelTime);

    // Calculate kernel execution time
    profile->kernelDuration = cutGetTimerValue(profile->kernelTime);

    // Check successfull execution of the kernel
    profile->kernelExecErr = cudaPeekAtLastError();
}

/*! Instantiates cu_Constant_Add_1D_Array_Impl() with the explicitly specified template for
 * input vector of type char.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput
 *          Input device vector.
 *
 * @param constVal
 *          Constant value to be added to the input device vector.
 *
 * @param devArrayOutput
 *          Output device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Constant_Add_1D_Array_Impl <char>
(dim3 cuBlock, dim3 cuGrid,
char *devArrayInput, char constVal, char* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Constant_Add_1D_Array_Impl() with the explicitly specified template for
 * input vector of type char.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput
 *          Input device vector.
 *
 * @param constVal
 *          Constant value to be added to the input device vector.
 *
 * @param devArrayOutput
 *          Output device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Constant_Add_1D_Array_Impl <unsigned char>
(dim3 cuBlock, dim3 cuGrid,
unsigned char* devArrayInput, unsigned char constVal, unsigned char* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Constant_Add_1D_Array_Impl() with the explicitly specified template for
 * input vector of type char.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput
 *          Input device vector.
 *
 * @param constVal
 *          Constant value to be added to the input device vector.
 *
 * @param devArrayOutput
 *          Output device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Constant_Add_1D_Array_Impl <int>
(dim3 cuBlock, dim3 cuGrid,
int* devArrayInput, int constVal, int* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Constant_Add_1D_Array_Impl() with the explicitly specified template for
 * input vector of type char.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput
 *          Input device vector.
 *
 * @param constVal
 *          Constant value to be added to the input device vector.
 *
 * @param devArrayOutput
 *          Output device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Constant_Add_1D_Array_Impl <unsigned int>
(dim3 cuBlock, dim3 cuGrid,
unsigned int* devArrayInput, unsigned int constVal, unsigned int* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Constant_Add_1D_Array_Impl() with the explicitly specified template for
 * input vector of type char.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput
 *          Input device vector.
 *
 * @param constVal
 *          Constant value to be added to the input device vector.
 *
 * @param devArrayOutput
 *          Output device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Constant_Add_1D_Array_Impl <float>
(dim3 cuBlock, dim3 cuGrid,
float* devArrayInput, float constVal, float* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Constant_Add_1D_Array_Impl() with the explicitly specified template for
 * input vector of type char.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput
 *          Input device vector.
 *
 * @param constVal
 *          Constant value to be added to the input device vector.
 *
 * @param devArrayOutput
 *          Output device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Constant_Add_1D_Array_Impl <double>
(dim3 cuBlock, dim3 cuGrid,
double* devArrayInput, double constVal, double* devArrayOutput, int N,
cuProfile* profile);

#endif // _CU_ADD_1D_ARRAY_DEVICE_IMPL_CU_
