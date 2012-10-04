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

#include "ex_Add_1D_Arrays.h"
#include "CUDA/cuGlobals.h"
#include "CUDA/cuUtilities.h"
#include "CUDA/cuYURI_Interfaces.h"
#include "Globals.h"
#include "Utilities/Utils.h"
#include "Utilities/MACROS.h"

#include "Utilities/MemoryMACROS.h"

void ex::Add_1D_Arrays::run(int argc, char* argv[])
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
    int *hostVectorOutput;

    // Device (GPU) vectors
    int *devInputVectorInput_1;
    int *devInputVectorInput_2;
    int *devInputVectorOutput;

    // Allocating the host array
    hostVectorOutput = MEM_ALLOC_1D(int, N);

    // Allocating device arrays
    devInputVectorInput_1 = cuUtils::Create_Device_Vector <int> (N);
    devInputVectorInput_2 = cuUtils::Create_Device_Vector <int> (N);
    devInputVectorOutput = cuUtils::Create_Device_Vector <int> (N);

    // Kernel configuration
    // NOTE: N should be power-of-two to have a SUCCESSFUL kernel execution
    dim3 cuBlock(4, 1, 1);
    dim3 cuGrid(N / cuBlock.x, 1, 1);

    // GPU profiler
    cuProfile profile;

    // Fill th input arrays on the device
    cuYURI::cu_Fill_1D_Array_RND <int>
            (cuBlock, cuGrid, devInputVectorInput_1, N, &profile);
    cuYURI::cu_Fill_1D_Array_RND <int>
            (cuBlock, cuGrid, devInputVectorInput_2, N, &profile);

    // Launch the kernel and get statistics
    cuYURI::cu_Add_1D_Arrays <int>
            (cuBlock, cuGrid,
             devInputVectorInput_1, devInputVectorInput_2, devInputVectorOutput,
             N, &profile);

    // Download the resulting vector to the host side
    cuUtils::Download_1D_Array <int> (hostVectorOutput, devInputVectorOutput, N);

    // Check the results
    INFO("Checking the resulting array on the host side")
    for (int i = 0; i < N; i++)
        INFO_LOOP(i, ITS(hostVectorOutput[i]));
    SEP();

    // Free host array
    FREE_MEM_1D(hostVectorOutput, int);

    // Free device memory
    cuUtils::Free_Device_Vector(devInputVectorOutput);

    // Application Finalization

    INFO("Done");
}
