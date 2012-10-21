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

namespace
{
template <typename T>
T* Create_Device_Vector(const int N);

template <typename T>
int Free_Device_Vector(T* deviceVecPtr);

template
char* Create_Device_Vector <char>
(const int N);

template
unsigned char* Create_Device_Vector <unsigned char>
(const int N);

template
int* Create_Device_Vector <int>
(const int N);

template
unsigned int* Create_Device_Vector <unsigned int>
(const int N);

template
float* Create_Device_Vector <float>
(const int N);

template
double* Create_Device_Vector <double>
(const int N);

template
cufftComplex* Create_Device_Vector <cufftComplex>
(const int N);

template
cufftDoubleComplex* Create_Device_Vector <cufftDoubleComplex>
(const int N);

template
int Free_Device_Vector <char>
(char* deviceVecPtr);

template
int Free_Device_Vector <unsigned char>
(unsigned char* deviceVecPtr);

template
int Free_Device_Vector <int>
(int* deviceVecPtr);

template
int Free_Device_Vector <unsigned int>
(unsigned int* deviceVecPtr);

template
int Free_Device_Vector <float>
(float* deviceVecPtr);

template
int Free_Device_Vector <double>
(double* deviceVecPtr);

template
int Free_Device_Vector <cufftComplex>
(cufftComplex* deviceVecPtr);

template
int Free_Device_Vector <cufftDoubleComplex>
(cufftDoubleComplex* deviceVecPtr);
}
