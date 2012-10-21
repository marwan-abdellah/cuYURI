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

#include "CUDA/Utilities/cuUtilities.h"

template <typename T>
T* cuUtils::Create_Device_Vector(const int N)
{
    LOG();

    INFO("Creating device vector");

    // Vector size in bytes
    const int vectorSize = N * sizeof(T);

    // Pointer to the device vector
    T* deviceVecPtr;

    // Allocate the device vector
    cudaMalloc((void**)(&deviceVecPtr), vectorSize);

    return deviceVecPtr;
}

template <typename T>
int cuUtils::Free_Device_Vector(T* deviceVecPtr)
{
    LOG();

    INFO("Freeing device memory");

    // Free device vector
    cudaFree(deviceVecPtr);

    return SUCCESS;
}

template
char* cuUtils::Create_Device_Vector <char>
(const int N);

template
unsigned char* cuUtils::Create_Device_Vector <unsigned char>
(const int N);

template
int* cuUtils::Create_Device_Vector <int>
(const int N);

template
unsigned int* cuUtils::Create_Device_Vector <unsigned int>
(const int N);

template
float* cuUtils::Create_Device_Vector <float>
(const int N);

template
double* cuUtils::Create_Device_Vector <double>
(const int N);

template
cufftComplex* cuUtils::Create_Device_Vector <cufftComplex>
(const int N);

template
cufftDoubleComplex* cuUtils::Create_Device_Vector <cufftDoubleComplex>
(const int N);

template
int cuUtils::Free_Device_Vector <char>
(char* deviceVecPtr);

template
int cuUtils::Free_Device_Vector <unsigned char>
(unsigned char* deviceVecPtr);

template
int cuUtils::Free_Device_Vector <int>
(int* deviceVecPtr);

template
int cuUtils::Free_Device_Vector <unsigned int>
(unsigned int* deviceVecPtr);

template
int cuUtils::Free_Device_Vector <float>
(float* deviceVecPtr);

template
int cuUtils::Free_Device_Vector <double>
(double* deviceVecPtr);

template
int cuUtils::Free_Device_Vector <cufftComplex>
(cufftComplex* deviceVecPtr);

template
int cuUtils::Free_Device_Vector <cufftDoubleComplex>
(cufftDoubleComplex* deviceVecPtr);
