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
 * CUDA : This kernel copies a source 1D array to a destination array
 * with length N
 *
 * @param devArray_Src
 *      Source device vector - or 1D array.
 *
 * @param devArray_Dist
 *      Destnation device vector - or 1D array.
 *
 * @param N
 *      Vector length.
 *
 * @author
 *      Marwan Abdellah <abdellah.marwan@gmail.com>
 *
 * @date
 *      Created: August, 2012.
 * @date
 *      Last Update: September, 2012.
 *
 * @note
 *      Minimum CUDA version 3.2.
 * @note
 *      Minimum Device Compute Capability 1.0.
 */
template <typename T>
__global__
void Copy_1D_Array_Kernel(T* devArray_Src, T* devArray_Dist, int N)
{
    int xThreadIdx = threadIdx.x;
    int blockWidth = blockDim.x;

    int index = blockIdx.x * blockWidth + xThreadIdx;

#ifdef VEC_CHECK
    if (index < N)
        devArray_Dist[index] = devArray_Src[index];
#else
    devArray_Dist[index] = devArray_Src[index];
#endif


}
