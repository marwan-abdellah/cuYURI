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

#include "ex_GPUConstantOperations.h"
#include "CUDA/cuGlobals.h"
#include "CUDA/Utilities/cuUtilities.h"
#include "CUDA/Utilities/cuKernelConf.h"
#include "CUDA/cuYURI_Interfaces.h"
#include "Globals.h"
#include "Utilities/Utils.h"
#include "MACROS/MACROS.h"
#include "MACROS/MemoryMACROS.h"

// For automatic kernel configuration generation
#define AUTO_CONF 1

void ex_GPUConstantOperations::Add(int argc, char* argv[])
{
    // Vector length N
    const int N = atoi(argv[1]);
    const double constVal = atoi(argv[2]);

    INFO("Input vector has a length of : " + ITS(N) +
         ", Const. value : " + DTS(constVal));

    // Host (CPU) vector (1D array)
    float *hostVectorOutput;

    // Device (GPU) vectors
    float *devInputVectorInput;
    float *devInputVectorOutput;

    // Allocating the host array
    hostVectorOutput = MEM_ALLOC_1D(float, N);

    // Allocating device arrays
    devInputVectorInput = cuUtils::Create_Device_Vector <float> (N);
    devInputVectorOutput = cuUtils::Create_Device_Vector <float> (N);

#ifdef AUTO_CONF
    // Generate optimized AUTOMATIC kernel configuration for the GPU
    kernelConf* autoConf = cuUtils::AutoGenKernelConf_1D(N);
#else
    // Kernel configuration
    // NOTE: N should be power-of-two to have a SUCCESSFUL kernel execution
    // for the manual configuration
    dim3 cuBlock(2, 1, 1);
    dim3 cuGrid(N / cuBlock.x, 1, 1);
#endif

    // GPU profiler
    cuProfile profile;

    // Fill th input array on the device
    cuYURI::cu_Fill_1D_Array <float>
            (autoConf->cuBlock, autoConf->cuGrid, devInputVectorInput, N, 0, &profile);

    // Display profiling data
    cuUtils::DisplayKernelProfilingData("cu_Fill_1D_Array", &profile);

    // Launch the kernel and get statistics
    cuYURI::cu_Constant_Add_1D_Array <float>
            (autoConf->cuBlock, autoConf->cuGrid,
             devInputVectorInput, constVal, devInputVectorOutput,
             N, &profile);

    // Download the resulting vector to the host side
    cuUtils::Download_1D_Array <float>
            (hostVectorOutput, devInputVectorOutput, N);

    // Display profiling data
    cuUtils::DisplayKernelProfilingData("cu_Constant_Add_1D_Array", &profile);

    // Check the results
    INFO("Checking the resulting array on the host side");
    for (int i = 0; i < N; i++)
        INFO_LOOP(i, FTS(hostVectorOutput[i]));
    SEP();

    // Free host array
    // FREE_MEM_1D(hostVectorOutput, int);

    // Free device memory
    cuUtils::Free_Device_Vector(devInputVectorInput);
    cuUtils::Free_Device_Vector(devInputVectorOutput);

    INFO("Done");
}

void ex_GPUConstantOperations::Subtract(int argc, char* argv[])
{
    // Vector length N
    const int N = atoi(argv[1]);
    const double constVal = atoi(argv[2]);

    INFO("Input vector has a length of : " + ITS(N) +
         ", Const. value : " + DTS(constVal));

    // Host (CPU) vector (1D array)
    float *hostVectorOutput;

    // Device (GPU) vectors
    float *devInputVectorInput;
    float *devInputVectorOutput;

    // Allocating the host array
    hostVectorOutput = MEM_ALLOC_1D(float, N);

    // Allocating device arrays
    devInputVectorInput = cuUtils::Create_Device_Vector <float> (N);
    devInputVectorOutput = cuUtils::Create_Device_Vector <float> (N);

#ifdef AUTO_CONF
    // Generate optimized AUTOMATIC kernel configuration for the GPU
    kernelConf* autoConf = cuUtils::AutoGenKernelConf_1D(N);
#else
    // Kernel configuration
    // NOTE: N should be power-of-two to have a SUCCESSFUL kernel execution
    // for the manual configuration
    dim3 cuBlock(2, 1, 1);
    dim3 cuGrid(N / cuBlock.x, 1, 1);
#endif

    // GPU profiler
    cuProfile profile;

    // Fill th input array on the device
    cuYURI::cu_Fill_1D_Array <float>
            (autoConf->cuBlock, autoConf->cuGrid, devInputVectorInput, N, 0, &profile);

    // Display profiling data
    cuUtils::DisplayKernelProfilingData("cu_Fill_1D_Array", &profile);

    // Launch the kernel and get statistics
    cuYURI::cu_Constant_Subtract_1D_Array <float>
            (autoConf->cuBlock, autoConf->cuGrid,
             devInputVectorInput, constVal, devInputVectorOutput,
             N, &profile);

    // Download the resulting vector to the host side
    cuUtils::Download_1D_Array <float>
            (hostVectorOutput, devInputVectorOutput, N);

    // Display profiling data
    cuUtils::DisplayKernelProfilingData("cu_Constant_Subtract_1D_Array", &profile);

    // Check the results
    INFO("Checking the resulting array on the host side");
    for (int i = 0; i < N; i++)
        INFO_LOOP(i, FTS(hostVectorOutput[i]));
    SEP();

    // Free host array
    // FREE_MEM_1D(hostVectorOutput, int);

    // Free device memory
    cuUtils::Free_Device_Vector(devInputVectorInput);
    cuUtils::Free_Device_Vector(devInputVectorOutput);

    INFO("Done");
}

void ex_GPUConstantOperations::Multiply(int argc, char* argv[])
{
    // Vector length N
    const int N = atoi(argv[1]);
    const double constVal = atoi(argv[2]);

    INFO("Input vector has a length of : " + ITS(N) +
         ", Const. value : " + DTS(constVal));

    // Host (CPU) vector (1D array)
    float *hostVectorOutput;

    // Device (GPU) vectors
    float *devInputVectorInput;
    float *devInputVectorOutput;

    // Allocating the host array
    hostVectorOutput = MEM_ALLOC_1D(float, N);

    // Allocating device arrays
    devInputVectorInput = cuUtils::Create_Device_Vector <float> (N);
    devInputVectorOutput = cuUtils::Create_Device_Vector <float> (N);

#ifdef AUTO_CONF
    // Generate optimized AUTOMATIC kernel configuration for the GPU
    kernelConf* autoConf = cuUtils::AutoGenKernelConf_1D(N);
#else
    // Kernel configuration
    // NOTE: N should be power-of-two to have a SUCCESSFUL kernel execution
    // for the manual configuration
    dim3 cuBlock(2, 1, 1);
    dim3 cuGrid(N / cuBlock.x, 1, 1);
#endif

    // GPU profiler
    cuProfile profile;

    // Fill th input array on the device
    cuYURI::cu_Fill_1D_Array <float>
            (autoConf->cuBlock, autoConf->cuGrid, devInputVectorInput, N, 0, &profile);

    // Display profiling data
    cuUtils::DisplayKernelProfilingData("cu_Fill_1D_Array", &profile);

    // Launch the kernel and get statistics
    cuYURI::cu_Constant_Multiply_1D_Array <float>
            (autoConf->cuBlock, autoConf->cuGrid,
             devInputVectorInput, constVal, devInputVectorOutput,
             N, &profile);

    // Download the resulting vector to the host side
    cuUtils::Download_1D_Array <float>
            (hostVectorOutput, devInputVectorOutput, N);

    // Display profiling data
    cuUtils::DisplayKernelProfilingData("cu_Constant_Multiply_1D_Array", &profile);

    // Check the results
    INFO("Checking the resulting array on the host side");
    for (int i = 0; i < N; i++)
        INFO_LOOP(i, FTS(hostVectorOutput[i]));
    SEP();

    // Free host array
    // FREE_MEM_1D(hostVectorOutput, int);

    // Free device memory
    cuUtils::Free_Device_Vector(devInputVectorInput);
    cuUtils::Free_Device_Vector(devInputVectorOutput);

    INFO("Done");
}

void ex_GPUConstantOperations::Divide(int argc, char* argv[])
{
    // Vector length N
    const int N = atoi(argv[1]);
    const double constVal = atoi(argv[2]);

    INFO("Input vector has a length of : " + ITS(N) +
         ", Const. value : " + DTS(constVal));

    // Host (CPU) vector (1D array)
    float *hostVectorOutput;

    // Device (GPU) vectors
    float *devInputVectorInput;
    float *devInputVectorOutput;

    // Allocating the host array
    hostVectorOutput = MEM_ALLOC_1D(float, N);

    // Allocating device arrays
    devInputVectorInput = cuUtils::Create_Device_Vector <float> (N);
    devInputVectorOutput = cuUtils::Create_Device_Vector <float> (N);

#ifdef AUTO_CONF
    // Generate optimized AUTOMATIC kernel configuration for the GPU
    kernelConf* autoConf = cuUtils::AutoGenKernelConf_1D(N);
#else
    // Kernel configuration
    // NOTE: N should be power-of-two to have a SUCCESSFUL kernel execution
    // for the manual configuration
    dim3 cuBlock(2, 1, 1);
    dim3 cuGrid(N / cuBlock.x, 1, 1);
#endif

    // GPU profiler
    cuProfile profile;

    // Fill th input array on the device
    cuYURI::cu_Fill_1D_Array <float>
            (autoConf->cuBlock, autoConf->cuGrid, devInputVectorInput, N, 0, &profile);

    // Display profiling data
    cuUtils::DisplayKernelProfilingData("cu_Fill_1D_Array", &profile);

    // Launch the kernel and get statistics
    cuYURI::cu_Constant_Divide_1D_Array <float>
            (autoConf->cuBlock, autoConf->cuGrid,
             devInputVectorInput, constVal, devInputVectorOutput,
             N, &profile);

    // Download the resulting vector to the host side
    cuUtils::Download_1D_Array <float>
            (hostVectorOutput, devInputVectorOutput, N);

    // Display profiling data
    cuUtils::DisplayKernelProfilingData("cu_Constant_Divide_1D_Array", &profile);

    // Check the results
    INFO("Checking the resulting array on the host side");
    for (int i = 0; i < N; i++)
        INFO_LOOP(i, FTS(hostVectorOutput[i]));
    SEP();

    // Free host array
    // FREE_MEM_1D(hostVectorOutput, int);

    // Free device memory
    cuUtils::Free_Device_Vector(devInputVectorInput);
    cuUtils::Free_Device_Vector(devInputVectorOutput);

    INFO("Done");
}

void ex_GPUConstantOperations::run(int argc, char* argv[])
{
    if (argv[1] == NULL || argv[2] == NULL)
    {
        INFO("Usage: ex_GPUConstantOperations <VECTRO_SIZE> <CONST_VALUE>");
        EXIT(0);
    }

    ex_GPUConstantOperations::Add(argc, argv);
    ex_GPUConstantOperations::Subtract(argc, argv);
    ex_GPUConstantOperations::Multiply(argc, argv);
    ex_GPUConstantOperations::Divide(argc, argv);
}
