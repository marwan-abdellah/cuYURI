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

#ifndef _MEMORY_3D_H_
#define _MEMORY_3D_H_

#include <cstdlib>
#include <fftw3.h>
#include <cufft.h>

namespace Memory
{
template <typename T>
extern
T*** Alloc_3D_Array(const int NX, const int NY, const int NZ);

template <typename T>
extern
void Free_3D_Array(T*** ptrData, const int NX, const int NY, int NZ);

template
char*** Alloc_3D_Array <char>
(const int , const int NY, const int NZ);

template
unsigned char*** Alloc_3D_Array <unsigned char>
(const int , const int NY, const int NZ);

template
int*** Alloc_3D_Array <int>
(const int NX, const int NY, const int NZ);

template
unsigned int*** Alloc_3D_Array <unsigned int>
(const int NX, const int NY, const int NZ);

template
float*** Alloc_3D_Array <float>
(const int NX, const int NY, const int NZ);

template
double*** Alloc_3D_Array <double>
(const int NX, const int NY, const int NZ);

template
fftwf_complex*** Alloc_3D_Array <fftwf_complex>
(const int NX, const int NY, const int NZ);

template
fftw_complex*** Alloc_3D_Array <fftw_complex>
(const int NX, const int NY, const int NZ);

template
cufftComplex*** Alloc_3D_Array <cufftComplex>
(const int NX, const int NY, const int NZ);

template
cufftDoubleComplex*** Alloc_3D_Array <cufftDoubleComplex>
(const int NX, const int NY, const int NZ);

template
void Free_3D_Array <char>
(char*** ptrData, const int NX, const int NY, int NZ);

template
void Free_3D_Array <unsigned char>
(unsigned char*** ptrData, const int NX, const int NY, int NZ);

template
void Free_3D_Array <int>
(int*** ptrData, const int NX, const int NY, int NZ);

template
void Free_3D_Array <unsigned int>
(unsigned int*** ptrData, const int NX, const int NY, int NZ);

template
void Free_3D_Array <float>
(float*** ptrData, const int NX, const int NY, int NZ);

template
void Free_3D_Array <double>
(double*** ptrData, const int NX, const int NY, int NZ);

template
void Free_3D_Array <fftwf_complex>
(fftwf_complex*** ptrData, const int NX, const int NY, int NZ);

template
void Free_3D_Array <fftw_complex>
(fftw_complex*** ptrData, const int NX, const int NY, int NZ);

template
void Free_3D_Array <cufftComplex>
(cufftComplex*** ptrData, const int NX, const int NY, int NZ);

template
void Free_3D_Array <cufftDoubleComplex>
(cufftDoubleComplex*** ptrData, const int NX, const int NY, int NZ);
}

#endif // _MEMORY_3D_H_
