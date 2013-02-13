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

#include "Memory_2D.h"

template <typename T>
T** Memory::Alloc_2D_Array(const int NX, const int NY)
{
    T** data;
    data = (T**) malloc (sizeof(T*) * NX);
    for(int i = 0; i < NX; i++)
        data[i] = (T*) malloc (sizeof(T) * NY);

    return data;
}

template <typename T>
extern
void Memory::Free_2D_Array(T** ptrData, const int NX, const int NY)
{
    for (int i = 0; i < NY; i++)
        free(ptrData[i]);

    free(ptrData);
    ptrData = NULL;
}

template
char** Memory::Alloc_2D_Array <char>
(const int NX, const int NY);

template
unsigned char** Memory::Alloc_2D_Array <unsigned char>
(const int NX, const int NY);

template
int** Memory::Alloc_2D_Array <int>
(const int NX, const int NY);

template
unsigned int** Memory::Alloc_2D_Array <unsigned int>
(const int NX, const int NY);

template
float** Memory::Alloc_2D_Array <float>
(const int NX, const int NY);

template
double** Memory::Alloc_2D_Array <double>
(const int NX, const int NY);

template
fftwf_complex** Memory::Alloc_2D_Array <fftwf_complex>
(const int NX, const int NY);

template
fftw_complex** Memory::Alloc_2D_Array <fftw_complex>
(const int NX, const int NY);

template
cufftComplex** Memory::Alloc_2D_Array <cufftComplex>
(const int NX, const int NY);

template
cufftDoubleComplex** Memory::Alloc_2D_Array
<cufftDoubleComplex> (const int NX, const int NY);

template
void Memory::Free_2D_Array <char>
(char** ptrData, const int NX, const int NY);

template
void Memory::Free_2D_Array <unsigned char>
(unsigned char** ptrData, const int NX, const int NY);

template
void Memory::Free_2D_Array <int>
(int** ptrData, const int NX, const int NY);

template
void Memory::Free_2D_Array <unsigned int>
(unsigned int** ptrData, const int NX, const int NY);

template
void Memory::Free_2D_Array <float>
(float** ptrData, const int NX, const int NY);

template
void Memory::Free_2D_Array <double>
(double** ptrData, const int NX, const int NY);

template
void Memory::Free_2D_Array <fftwf_complex>
(fftwf_complex** ptrData, const int NX, const int NY);

template
void Memory::Free_2D_Array <fftw_complex>
(fftw_complex** ptrData, const int NX, const int NY);

template
void Memory::Free_2D_Array <cufftComplex>
(cufftComplex** ptrData, const int NX, const int NY);

template
void Memory::Free_2D_Array <cufftDoubleComplex>
(cufftDoubleComplex** ptrData, const int NX, const int NY);
