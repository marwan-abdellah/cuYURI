/*********************************************************************
 * Copyright © 2011-2013,
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

#ifndef _MEMORY_1D_H_
#define _MEMORY_1D_H_

#include <cstdlib>
#include <fftw3.h>
#include <cufft.h>

namespace Memory
{
template <typename T>
extern
T* Alloc_1D_Array(const int NX);

template <typename T>
extern
void Free_1D_Array(T* ptrData);

template
char* Alloc_1D_Array <char>
(const int NX);

template
unsigned char* Alloc_1D_Array <unsigned char>
(const int NX);

template
int* Alloc_1D_Array <int>
(const int NX);

template
unsigned int* Alloc_1D_Array <unsigned int>
(const int NX);

template
float* Alloc_1D_Array <float>
(const int NX);

template
double* Alloc_1D_Array <double>
(const int NX);

template
fftwf_complex* Alloc_1D_Array <fftwf_complex>
(const int NX);

template
fftw_complex* Alloc_1D_Array <fftw_complex>
(const int NX);

template
cufftComplex* Alloc_1D_Array <cufftComplex>
(const int NX);

template
cufftDoubleComplex* Alloc_1D_Array <cufftDoubleComplex>
(const int NX);

template
void Free_1D_Array <char>
(char* ptrData);

template
void Free_1D_Array <unsigned char>
(unsigned char* ptrData);

template
void Free_1D_Array <int>
(int* ptrData);

template
void Free_1D_Array <unsigned int>
(unsigned int* ptrData);

template
void Free_1D_Array <float>
(float* ptrData);

template
void Free_1D_Array <double>
(double* ptrData);

template
void Free_1D_Array <fftwf_complex>
(fftwf_complex* ptrData);

template
void Free_1D_Array <fftw_complex>
(fftw_complex* ptrData);

template
void Free_1D_Array <cufftComplex>
(cufftComplex* ptrData);

template
void Free_1D_Array <cufftDoubleComplex>
(cufftDoubleComplex* ptrData);
}

#endif // _MEMORY_1D_H_
