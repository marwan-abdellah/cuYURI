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
 * CUDA : This kernel multiplies 2 input vectors to the output vector of length N
 *
 * @param devArrayInput_1
 *      First input array.
 *
 * @param devArrayInput_2
 *      Second input vector.
 *
 * @param devArrayOutput
 *      Resulting vector.
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
void Multiply_1D_Arrays_Kernel(T* devArrayInput_1,
                               T* devArrayInput_2,
                               T* devArrayOutput,
                               int N)
{
    int xThreadIdx = threadIdx.x;
    int blockWidth = blockDim.x;

    int index = blockIdx.x * blockWidth + xThreadIdx;

#ifdef VEC_CHECK
    if (index < N)
        devArrayOutput[index] = devArrayInput_1[index] * devArrayInput_2[index];
#else
    devArrayOutput[index] = devArrayInput_1[index] * devArrayInput_2[index];
#endif


}
