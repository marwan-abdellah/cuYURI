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
 * CUDA : This kernel adds a constant value to the input vector into the
 * output vector with length N.
 *
 * @param devArrayInput
 *      Input vector.
 *
 * @param constVal
 *      Constant value to be added to the input device vector.
 *
 * @param devArrayOutput
 *      Sum vector.
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
void Constant_Add_1D_Array_Kernel(T* devArrayInput,
                                 T constVal,
                                 T* devArrayOutput,
                                 int N)
{
    int xThreadIdx = threadIdx.x;
    int blockWidth = blockDim.x;

    int index = blockIdx.x * blockWidth + xThreadIdx;

#ifdef VEC_CHECK
    if (index < N)
        devArrayOutput[index] = (T) ((T) devArrayInput[index] + (T) constVal);
#else
    devArrayOutput[index] = (T) ((T) devArrayInput[index] + (T) constVal);
#endif
}
