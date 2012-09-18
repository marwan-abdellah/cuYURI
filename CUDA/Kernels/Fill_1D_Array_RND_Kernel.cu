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

#include "cuGlobals.h"

/*!
 * CUDA : This kernel fills an input vector - or 1D array - with random
 * sequence of numbers of length N.
 *
 * @param devArray
 *      Input device vector - or 1D array - to the kernel.
 *
 * @param N
 *      Vector length.
 *
 * @author
 *      Marwan Abdellah <abdellah.marwan@gmail.com>
 *
 * @date
 *      Created: April, 2011.
 * @date
 *      Last Update: September, 2012.
 *
 * @note
 *      Minimum CUDA version 3.2.
 * @note
 *      Device Compute Capability 1.0.
 */
template <typename T>
__global__
void Fill_1D_Array_RND_Kernel(T* devArray, int N)
{
    // Thread index @X
    int x_threadIdx = threadIdx.x;

    // Block size @X
    int x_blockDim = blockDim.x;

    // Block index @X
    int x_blockIdx = blockIdx.x;

    // Thread flat index
    int index = ((x_blockIdx * x_blockDim) + x_threadIdx);

#ifdef VEC_CHECK
    if (index < N)
        devArray[index] = index;
#else
    devArray[index] = (T) index;
#endif
}
