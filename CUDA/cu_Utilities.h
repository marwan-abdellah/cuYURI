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

#ifndef CUUTILS_H_
#define CUUTILS_H_

#include "CUDA/cu_Globals.h"
#include "Utilities/Utils.h"
#include "Globals.h"
#include "Utilities/Logging.h"
#include "Utilities/MACROS.h"

namespace cuUtils
{
    template <typename T>
    int Upload_1D_Array(T* hostArray, T* deviceArray, int N);

    template <typename T>
    int Download_1D_Array(T* hostArray, T* deviceArray, int N);

    template <typename T>
    T* Create_Device_Vector(const int N);

    template <typename T>
    int Free_Device_Vector(T* deviceVecPtr);

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







	int upload_1D_float(float* hostArr, float* devArr, int size_X);
	int upload_2D_float(float* hostArr, float* devArr, int size_X, int size_Y);
	int upload_3D_float(float* hostArr, float* devArr, int size_X, int size_Y, int size_Z);

	int download_1D_float(float* hostArr, float* devArr, int size_X);
    int download_1D_int(int* hostArr, int* devArr, int size_X);
	int download_2D_float(float* hostArr, float* devArr, int size_X, int size_Y);
	int download_3D_float(float* hostArr, float* devArr, int size_X, int size_Y, int size_Z);

	int upload_1D_double(double* hostArr, double* devArr, int size_X);
	int upload_2D_double(double* hostArr, double* devArr, int size_X, int size_Y);
	int upload_3D_double(double* hostArr, double* devArr, int size_X, int size_Y, int size_Z);

	int download_1D_double(double* hostArr, double* devArr, int size_X);
	int download_2D_double(double* hostArr, double* devArr, int size_X, int size_Y);
	int download_3D_double(double* hostArr, double* devArr, int size_X, int size_Y, int size_Z);

	int upload_1D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X);
	int upload_2D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y);
	int upload_3D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y, int size_Z);

	int download_1D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X);
	int download_2D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y);
	int download_3D_cuComplex(cufftComplex* hostArr, cufftComplex* devArr, int size_X, int size_Y, int size_Z);

	int upload_1D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X);
	int upload_2D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y);
	int upload_3D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y, int size_Z);

	int download_1D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X);
	int download_2D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y);
	int download_3D_cuDoubleComplex(cufftDoubleComplex* hostArr, cufftDoubleComplex* devArr, int size_X, int size_Y, int size_Z);
}

#endif /* CUUTILS_H_ */
