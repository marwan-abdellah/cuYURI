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

#include "ex_FillArrayRND.h"
#include "CUDA/cuGlobals.h"
#include "CUDA/Utilities/cuUtilities.h"
#include "CUDA/cuYURI_Interfaces.h"
#include "Globals.h"
#include "Utilities/Utils.h"
#include "Utilities/MACROS.h"
#include "Utilities/MemoryMACROS.h"

// Sets the kernel GPU configuration automatically
#define AUTO_CONF 1

void ex_FillArrayRND::run(int argc, char* argv[])
{
    if (argv[1] == NULL)
    {
        INFO("Usage: ex_FillArrayRND <VECTRO_SIZE>");
        EXIT(0);
    }

    // Vector length N
    const int N = atoi(argv[1]);

    INFO("Input vector has a length of " + ITS(N));

    // Host (CPU) vector (1D array)
    float *hostVector;

    // Device (GPU) vector
    float *deviceVector;

    // Allocating the host array
    hostVector = MEM_ALLOC_1D(float, N);

    // Allocating device array
    deviceVector = cuUtils::Create_Device_Vector <float> (N);

    // GPU profiler
    cuProfile profile;

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

    // Launch the kernel and get statistics
    cuYURI::cu_Fill_1D_Array_RND <float> (autoConf->cuBlock,
                                          autoConf->cuGrid,
                                          deviceVector, N, &profile);

    // Display profiling data
    cuUtils::DisplayKernelProfilingData("cu_Fill_1D_Array_RND", &profile);

    // Download the resulting vector to the host side
    cuUtils::Download_1D_Array <float> (hostVector, deviceVector, N);

    // Check the results
    INFO("Checking the resulting array on the host side")
   // for (int i = 0; i < N; i++)
       // INFO_LOOP(i, FTS(hostVector[i]));
    SEP();

    // Free host array
    FREE_MEM_1D(hostVector, float);

    // Free device memory
    cuUtils::Free_Device_Vector(deviceVector);

    INFO("Done");
}
