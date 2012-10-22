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
 * \file Constant_Divide_1D_Array.h
 *
 * @brief This interface links directly to the implementation of the
 *      Constant_Divide_1D_Array_Device_Kernel.
 *
 * @author
 *      Marwan Abdellah <abdellah.marwan@gmail.com>
 *
 * @date
 *      Created: August, 2011.
 * @date
 *      Last Update: September, 2012.
 */

#ifndef _CONSTANT_DIVIDE_1D_ARRAY_DEVICE_H_
#define _CONSTANT_DIVIDE_1D_ARRAY_DEVICE_H_

template <typename T>
extern
void cu_Constant_Divide_1D_Array_Impl
(dim3 cuBlock, dim3 cuGrid,
 T* devArrayInput,  T constVal, T* devArrayOutput, int N,
 cuProfile* profile);

namespace cuYURI
{
template <typename T>
extern
void cu_Constant_Divide_1D_Array
(dim3 cuBlock, dim3 cuGrid,
 T* devArrayInput,  T constVal, T* devArrayOutput, int N,
 cuProfile* profile)
{
    cu_Constant_Divide_1D_Array_Impl(cuBlock, cuGrid, devArrayInput, constVal, devArrayOutput, N, profile);
}

/*! Instantiates cu_Constant_Divide_1D_Array() with the explicitly specified template for
 * input vector of type char.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput
 *          Input device vector.
 *
 * @param constVal
 *          Constant value to be have the input device vector divided by.
 *
 * @param devArrayOutput
 *          Output device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Constant_Divide_1D_Array <char>
(dim3 cuBlock, dim3 cuGrid,
char* devArrayInput, char constVal,
char* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Constant_Divide_1D_Array() with the explicitly specified template for
 * input vector of type unsigned char.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput
 *          Input device vector.
 *
 * @param constVal
 *          Constant value to be have the input device vector divided by.
 *
 * @param devArrayOutput
 *          Output device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Constant_Divide_1D_Array <unsigned char>
(dim3 cuBlock, dim3 cuGrid,
unsigned char* devArrayInput, unsigned char constVal,
unsigned char* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Constant_Divide_1D_Array() with the explicitly specified template for
 * input vector of type int.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput
 *          Input device vector.
 *
 * @param constVal
 *          Constant value to be have the input device vector divided by.
 *
 * @param devArrayOutput
 *          Output device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Constant_Divide_1D_Array <int>
(dim3 cuBlock, dim3 cuGrid,
int* devArrayInput, int constVal, int* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Constant_Divide_1D_Array() with the explicitly specified template for
 * input vector of type unsigned int.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput
 *          Input device vector.
 *
 * @param constVal
 *          Constant value to be have the input device vector divided by.
 *
 * @param devArrayOutput
 *          Output device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Constant_Divide_1D_Array <unsigned int>
(dim3 cuBlock, dim3 cuGrid,
unsigned int* devArrayInput, unsigned int constVal,
unsigned int* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Constant_Divide_1D_Array() with the explicitly specified template for
 * input vector of type float.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput
 *          Input device vector.
 *
 * @param constVal
 *          Constant value to be have the input device vector divided by.
 *
 * @param devArrayOutput
 *          Output device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Constant_Divide_1D_Array <float>
(dim3 cuBlock, dim3 cuGrid,
float* devArrayInput, float constVal,
float* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Constant_Divide_1D_Array() with the explicitly specified template for
 * input vector of type double.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput
 *          Input device vector.
 *
 * @param constVal
 *          Constant value to be have the input device vector divided by.
 *
 * @param devArrayOutput
 *          Output device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Constant_Divide_1D_Array <double>
(dim3 cuBlock, dim3 cuGrid,
double* devArrayInput, double constVal,
double* devArrayOutput, int N,
cuProfile* profile);
}
#endif // _CONSTANT_DIVIDE_1D_ARRAY_DEVICE_H_
