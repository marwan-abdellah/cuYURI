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
 * \file Add_1D_Arrays.h
 *
 * @brief This interface links directly to the implementation of the
 *      Add_1D_Arrays_Device_Kernel.
 *
 * @author
 *      Marwan Abdellah <abdellah.marwan@gmail.com>
 *
 * @date
 *      Created: August, 2011.
 * @date
 *      Last Update: September, 2012.
 */

#ifndef _ADD_1D_ARRAYS_DEVICE_H_
#define _ADD_1D_ARRAYS_DEVICE_H_

template <typename T>
extern
void cu_Add_1D_Arrays_Impl
(dim3 cuBlock, dim3 cuGrid,
 T *devArrayInput_1,  T* devArrayInput_2, T* devArrayOutput, int N,
 cuProfile* profile);

namespace cuYURI
{
template <typename T>
extern
void cu_Add_1D_Arrays
(dim3 cuBlock, dim3 cuGrid,
 T *devArrayInput_1,  T* devArrayInput_2, T* devArrayOutput, int N,
 cuProfile* profile)
{
    cu_Add_1D_Arrays_Impl(cuBlock, cuGrid, devArrayInput_1, devArrayInput_2, devArrayOutput, N, profile);
}

/*! Instantiates cu_Add_1D_Arrays() with the explicitly specified template for
 * input vector of type char.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput_1
 *          First input device vector.
 *
 * @param devArrayInput_2
 *          Second input device vector.
 *
 * @param devArrayOutput
 *          Output (Sum) device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Add_1D_Arrays <char>
(dim3 cuBlock, dim3 cuGrid,
char *devArrayInput_1, char *devArrayInput_2,
char* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Add_1D_Arrays() with the explicitly specified template for
 * input vector of type unsigned char.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput_1
 *          First input device vector.
 *
 * @param devArrayInput_2
 *          Second input device vector.
 *
 * @param devArrayOutput
 *          Output (Sum) device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Add_1D_Arrays <unsigned char>
(dim3 cuBlock, dim3 cuGrid,
unsigned char *devArrayInput_1, unsigned char *devArrayInput_2,
unsigned char* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Add_1D_Arrays() with the explicitly specified template for
 * input vector of type int.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput_1
 *          First input device vector.
 *
 * @param devArrayInput_2
 *          Second input device vector.
 *
 * @param devArrayOutput
 *          Output (Sum) device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Add_1D_Arrays <int>
(dim3 cuBlock, dim3 cuGrid,
int *devArrayInput_1, int *devArrayInput_2, int* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Add_1D_Arrays() with the explicitly specified template for
 * input vector of type unsigned int.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput_1
 *          First input device vector.
 *
 * @param devArrayInput_2
 *          Second input device vector.
 *
 * @param devArrayOutput
 *          Output (Sum) device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Add_1D_Arrays <unsigned int>
(dim3 cuBlock, dim3 cuGrid,
unsigned int *devArrayInput_1, unsigned int *devArrayInput_2,
unsigned int* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Add_1D_Arrays() with the explicitly specified template for
 * input vector of type float.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput_1
 *          First input device vector.
 *
 * @param devArrayInput_2
 *          Second input device vector.
 *
 * @param devArrayOutput
 *          Output (Sum) device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Add_1D_Arrays <float>
(dim3 cuBlock, dim3 cuGrid,
float *devArrayInput_1, float *devArrayInput_2,
float* devArrayOutput, int N,
cuProfile* profile);

/*! Instantiates cu_Add_1D_Arrays() with the explicitly specified template for
 * input vector of type double.
 *
 * @param cuBlock
 *          Kernel block configuration.
 *
 * @param cuGrid
 *          Kernel grid configuration.
 *
 * @param devArrayInput_1
 *          First input device vector.
 *
 * @param devArrayInput_2
 *          Second input device vector.
 *
 * @param devArrayOutput
 *          Output (Sum) device vector.
 *
 * @param N
 *          Length of the input vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 */
template
void cu_Add_1D_Arrays <double>
(dim3 cuBlock, dim3 cuGrid,
double *devArrayInput_1, double *devArrayInput_2,
double* devArrayOutput, int N,
cuProfile* profile);
}
#endif // _ADD_1D_ARRAYS_DEVICE_H_
