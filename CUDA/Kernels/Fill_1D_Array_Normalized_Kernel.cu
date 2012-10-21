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

/*!
 * CUDA : This kernel fills an input vector - or 1D array - with a
 * sequence of numbers of length N.
 *
 * @param devArray
 *      Input device vector - or 1D array - to the kernel.
 *
 * @param N
 *      Vector length.
 *
 * @param fillOrder
 *      Ascending if <0> or descending order <1>.
 *
 * @author
 *      Marwan Abdellah <abdellah.marwan@gmail.com>
 *
 * @date
 *      Created: June, 2011.
 * @date
 *      Last Update: October, 2012.
 *
 * @note
 *      Minimum CUDA version 3.2.
 * @note
 *      Minimum Device Compute Capability 1.0.
 */
template <typename T>
__global__
void Fill_1D_Array_Normalized_Kernel(T* devArray, int N, int fillOrder)
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
    {
        if (fillOrder == 0)
            devArray[index] = (T) ((float) index / (float) N);
        else
            devArray[index] = (T) ((float) (N - index) / (float) N);
    }
#else
    if (fillOrder == 0)
        devArray[index] = (T) ((float) index / (float) N);
    else
        devArray[index] = (T) ((float) (N - index) / (float) N);
#endif
}
