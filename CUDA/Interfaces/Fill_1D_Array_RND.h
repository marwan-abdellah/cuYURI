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

/**
 * \file Fill_1D_Array_RND.h
 *
 * @brief Includes all the mandatory interfaces and header files for
 * CUDA kernels and utility functions from the NVIDIA CUDA Toolkit
 * and the GPU Computing SDK.
 *
 * @author
 *      Marwan Abdellah <abdellah.marwan@gmail.com>
 *
 * @date
 *      Created: April, 2011.
 * @date
 *      Last Update: September, 2012.
 */
#ifndef _FILL_1D_ARRAY_RND_H_
#define _FILL_1D_ARRAY_RND_H_

template <typename T>
extern
void cu_Fill_1D_Array_RND_Impl
(dim3 cuBlock, dim3 cuGrid,
 T *devArray, int N,
 cuProfile* profile);

namespace cuYURI
{
template <typename T>
extern
void cu_Fill_1D_Array_RND
(dim3 cuBlock, dim3 cuGrid,
 T *devArray, int N,
 cuProfile* profile)
{
    cu_Fill_1D_Array_RND_Impl( cuBlock, cuGrid, devArray, N, profile);
}

/*! Instantiates cu_Fill_1D_Array_RND() with the explicitly specified template for
 * input vector of type char.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArray
 *          Input device vector to the kernel.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Fill_1D_Array_RND <char>
(dim3 cuBlock, dim3 cuGrid,
char *devArray, int N,
cuProfile* profile);

/*! Instantiates cu_Fill_1D_Array_RND() with the explicitly specified template for
 * input vector of type unsigned char.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArray
 *          Input device vector to the kernel.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Fill_1D_Array_RND <unsigned char>
(dim3 cuBlock, dim3 cuGrid,
unsigned char *devArray, int N,
cuProfile* profile);

/*! Instantiates cu_Fill_1D_Array_RND() with the explicitly specified template for
 * input vector of type int.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArray
 *          Input device vector to the kernel.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Fill_1D_Array_RND <int>
(dim3 cuBlock, dim3 cuGrid,
int *devArray, int N,
cuProfile* profile);

/*! Instantiates cu_Fill_1D_Array_RND() with the explicitly specified template for
 * input vector of type unsigned int.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArray
 *          Input device vector to the kernel.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Fill_1D_Array_RND <unsigned int>
(dim3 cuBlock, dim3 cuGrid,
unsigned int *devArray, int N,
cuProfile* profile);

/*! Instantiates cu_Fill_1D_Array_RND() with the explicitly specified template for
 * input vector of type float.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArray
 *          Input device vector to the kernel.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Fill_1D_Array_RND <float>
(dim3 cuBlock, dim3 cuGrid,
float *devArray, int N,
cuProfile* profile);

/*! Instantiates cu_Fill_1D_Array_RND() with the explicitly specified template for
 * input vector of type double.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArray
 *          Input device vector to the kernel.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Fill_1D_Array_RND <double>
(dim3 cuBlock, dim3 cuGrid,
double *devArray, int N,
cuProfile* profile);

} // namespace cuYURI

#endif // _FILL_1D_ARRAY_RND_H_
