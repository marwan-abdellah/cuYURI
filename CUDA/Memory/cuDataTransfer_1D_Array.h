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

namespace cuUtils
{
template <typename T>
int Upload_1D_Array(T* hostArray, T* deviceArray, int N);

template <typename T>
int Download_1D_Array(T* hostArray, T* deviceArray, int N);

template
int Upload_1D_Array <char>
(char* hostArray, char* deviceArray, int N);

template
int Upload_1D_Array <unsigned char>
(unsigned char* hostArray, unsigned char* deviceArray, int N);

template
int Upload_1D_Array <int>
(int* hostArray, int* deviceArray, int N);

template
int Upload_1D_Array <unsigned int>
(unsigned int* hostArray, unsigned int* deviceArray, int N);

template
int Upload_1D_Array <float>
(float* hostArray, float* deviceArray, int N);

template
int Upload_1D_Array <double>
(double* hostArray, double* deviceArray, int N);

template
int Upload_1D_Array <cufftComplex>
(cufftComplex* hostArray, cufftComplex* deviceArray, int N);

template
int Upload_1D_Array <cufftDoubleComplex>
(cufftDoubleComplex* hostArray, cufftDoubleComplex* deviceArray, int N);

template
int Download_1D_Array <char>
(char* hostArray, char* deviceArray, int N);

template
int Download_1D_Array <unsigned char>
(unsigned char* hostArray, unsigned char* deviceArray, int N);

template
int Download_1D_Array <int>
(int* hostArray, int* deviceArray, int N);

template
int Download_1D_Array <unsigned int>
(unsigned int* hostArray, unsigned int* deviceArray, int N);

template
int Download_1D_Array <float>
(float* hostArray, float* deviceArray, int N);

template
int Download_1D_Array <double>
(double* hostArray, double* deviceArray, int N);

template
int Download_1D_Array <cufftComplex>
(cufftComplex* hostArray, cufftComplex* deviceArray, int N);

template
int Download_1D_Array <cufftDoubleComplex>
(cufftDoubleComplex* hostArray, cufftDoubleComplex* deviceArray, int N);
}
