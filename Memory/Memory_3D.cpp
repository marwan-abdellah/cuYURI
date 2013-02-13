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

#include "Memory_3D.h"

template <typename T>
extern
T*** Memory::Alloc_3D_Array(const int NX, const int NY, const int NZ)
{
    T*** data;
    data = (T***) malloc (sizeof(T**) * NX);
    for(int i = 0; i < NX; i++)
    {
        data[i] = (T**) malloc (sizeof(T*) * NY);
        for (int j = 0; j < NY; j++)
            data[i][j] = (T*) malloc (sizeof(T) * NZ);
    }

    return data;
}

template <typename T>
extern
void Memory::Free_3D_Array(T*** ptrData, const int NX, const int NY, int NZ)
{
    for(int i = 0; i < NY; i++)
    {
        for(int j = 0; j < NZ; j++)
        {
            // Z
            free(ptrData[i][j]);
        }

        // Y
        free(ptrData[i]);
    }

    // X
    free(ptrData);
    ptrData = NULL;
}

template
char*** Memory::Alloc_3D_Array <char>
(const int , const int NY, const int NZ);

template
unsigned char*** Memory::Alloc_3D_Array <unsigned char>
(const int , const int NY, const int NZ);

template
int*** Memory::Alloc_3D_Array <int>
(const int NX, const int NY, const int NZ);

template
unsigned int*** Memory::Alloc_3D_Array <unsigned int>
(const int NX, const int NY, const int NZ);

template
float*** Memory::Alloc_3D_Array <float>
(const int NX, const int NY, const int NZ);

template
double*** Memory::Alloc_3D_Array <double>
(const int NX, const int NY, const int NZ);

template
fftwf_complex*** Memory::Alloc_3D_Array <fftwf_complex>
(const int NX, const int NY, const int NZ);

template
fftw_complex*** Memory::Alloc_3D_Array <fftw_complex>
(const int NX, const int NY, const int NZ);

template
cufftComplex*** Memory::Alloc_3D_Array <cufftComplex>
(const int NX, const int NY, const int NZ);

template
cufftDoubleComplex*** Memory::Alloc_3D_Array <cufftDoubleComplex>
(const int NX, const int NY, const int NZ);

template
void Memory::Free_3D_Array <char>
(char*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <unsigned char>
(unsigned char*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <int>
(int*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <unsigned int>
(unsigned int*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <float>
(float*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <double>
(double*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <fftwf_complex>
(fftwf_complex*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <fftw_complex>
(fftw_complex*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <cufftComplex>
(cufftComplex*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <cufftDoubleComplex>
(cufftDoubleComplex*** ptrData, const int NX, const int NY, int NZ);
