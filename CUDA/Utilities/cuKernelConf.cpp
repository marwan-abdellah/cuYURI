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

#include "cuKernelConf.h"
#include "MACROS/MACROS.h"

/*! Automatically detects the GPU configuration and generates
 * optimized kernel configuration matching the size of the
 * 1D vector \param N
 *
 * @param N
 *          Length of the device vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 * @return Automatically calculated CUDA kernel configuration
 *
 */
kernelConf* cuUtils::AutoGenKernelConf_1D(const int N)
{
    INFO("Detecting the best kernel configuration");

    // Automatic configuration for the kernel
    kernelConf* autoConf = MEM_ALLOC_1D_GENERIC(kernelConf, 1);

    int threadsPerBlock;

    // Set the threadsPerBlock parameter according to N
    if (2 <= N && N < 4)        threadsPerBlock = 2;
    if (4 <= N && N < 8)        threadsPerBlock = 4;
    if (8 <= N && N < 16)       threadsPerBlock = 8;
    if (16 <= N && N < 32)      threadsPerBlock = 16;
    if (32 <= N && N < 64)      threadsPerBlock = 32;
    if (64 <= N && N < 128)     threadsPerBlock = 64;
    if (128 <= N && N < 256)    threadsPerBlock = 128;
    if (256 <= N && N < 512)    threadsPerBlock = 256;
    if (N >= 512)               threadsPerBlock = 512;

    INFO("AutoKernelConf: THREADS_PER_BLOCK = " + ITS(threadsPerBlock));

    // Automatic kernel configuration according to N or the array length
    autoConf->cuBlock = dim3(threadsPerBlock, 1, 1);
    autoConf->cuGrid = dim3(N / threadsPerBlock, 1, 1);

    INFO("Generated kernel configuration is "
         "[" + ITS(autoConf->cuBlock.x) +
         "," + ITS(autoConf->cuBlock.y) +
         "," + ITS(autoConf->cuBlock.z) + "] Block & " +
         "[" + ITS(autoConf->cuGrid.x) +
         "," + ITS(autoConf->cuGrid.y) +
         "," +ITS(autoConf->cuGrid.z) + "] Grid");

    return autoConf;
}
