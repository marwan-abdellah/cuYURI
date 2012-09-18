/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File         : cuMath.cu
 * Author(s)    : Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created      : April 2011
 * Description  :
 * Note(s)      :
 *********************************************************************/

#ifndef CU_MATH_CU_
#define CU_MATH_CU_

#include <cutil_inline.h>
#include <cufft.h>



__global__
void add_Kernel(const int* devArrayInput_1, const int* devArrayInput_2,
                int* devArrayOutput,
                const int arraySize)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < arraySize)
        devArrayOutput[index] = devArrayInput_1[index] + devArrayInput_2[index];
}

#endif // CU_MATH_CU_
