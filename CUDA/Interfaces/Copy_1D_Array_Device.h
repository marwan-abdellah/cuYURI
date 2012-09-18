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

#ifndef _COPY_1D_ARRAY_DEVICE_H_
#define _COPY_1D_ARRAY_DEVICE_H_

template <typename T>
extern
void cu_Copy_1D_Array_Device_Impl
(dim3 cuBlock, dim3 cuGrid,
 T *devArray_Src, T* devArray_Dist, int N,
 cu_Profile* profile);

template <typename T>
extern
void cu_Copy_1D_Array_Device
(dim3 cuBlock, dim3 cuGrid,
 T *devArray_Src, T* devArray_Dist, int N,
 cu_Profile* profile)
{
    cu_Copy_1D_Array_Device_Impl( cuBlock, cuGrid, devArray, N, profile);
}

template
void cu_Copy_1D_Array_Device <char>
(dim3 cuBlock, dim3 cuGrid,
char *devArray_Src, char* devArray_Dist, int N,
cu_Profile* profile);

template
void cu_Copy_1D_Array_Device <unsigned char>
(dim3 cuBlock, dim3 cuGrid,
unsigned char *devArray_Src, unsigned char* devArray_Dist, int N,
cu_Profile* profile);

template
void cu_Copy_1D_Array_Device <int>
(dim3 cuBlock, dim3 cuGrid,
int *devArray_Src, int* devArray_Dist, int N,
cu_Profile* profile);

template
void cu_Copy_1D_Array_Device <unsigned int>
(dim3 cuBlock, dim3 cuGrid,
unsigned int *devArray_Src, unsigned int* devArray_Dist, int N,
cu_Profile* profile);

template
void cu_Copy_1D_Array_Device <float>
(dim3 cuBlock, dim3 cuGrid,
float *devArray_Src, float* devArray_Dist, int N,
cu_Profile* profile);

template
void cu_Copy_1D_Array_Device <double>
(dim3 cuBlock, dim3 cuGrid,
double *devArray_Src, double* devArray_Dist, int N,
cu_Profile* profile);

#endif // _COPY_1D_ARRAY_DEVICE_H_
