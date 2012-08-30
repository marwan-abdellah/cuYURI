 /*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File         : cuExternsTest.cu
 * Author(s)    : Marwan Abdellah   <abdellah.marwan@gmail.com> 
		  Salah Saleh	    <salah_saleh@live.com>	  
 * Created      : July 2011
 * Description  :
 * Note(s)      :
 *********************************************************************/
 
#ifndef CU_EXTERNS_TEST_CU_
#define CU_EXTERNS_TEST_CU_

#include "CUDA/cuGlobals.h"

#include <cutil_inline.h>

#include "MaxSubArray.cu"
#include "Timers/BoostTimers.h"
#include "Shared.h"

extern  
void cuPrefixSum(dim3 cuBlock, dim3 cuGrid, 
              int* devInputArray, int* devPrefixSum, 
              int numRows, int numCols, cudaProfile* cuProfile)
{
    cutCreateTimer(&(cuProfile->kernelTime));
    cutResetTimer(cuProfile->kernelTime);
    cutStartTimer(cuProfile->kernelTime);
    
    prefixSum <<<cuGrid, cuBlock>>> (numRows, numCols, devInputArray, devPrefixSum); 
    cudaThreadSynchronize(); 
    
    cutStopTimer(cuProfile->kernelTime);
    
    cuProfile->kernelDuration = cutGetTimerValue(cuProfile->kernelTime);
    cuProfile->kernelExecErr = cudaPeekAtLastError();
}

extern  
void cuGetMax(dim3 cuBlock, dim3 cuGrid, 
              Max* devMaxValues, int* devPrefixSum, 
              int numRows, int numCols, cudaProfile* cuProfile)
{
    cutCreateTimer(&(cuProfile->kernelTime));
    cutResetTimer(cuProfile->kernelTime);
    cutStartTimer(cuProfile->kernelTime);
    
    findMax <<<cuGrid, cuBlock>>> (numRows, numCols, devMaxValues, devPrefixSum); 
    cudaThreadSynchronize(); 
    
    cutStopTimer(cuProfile->kernelTime);
    
    cuProfile->kernelDuration = cutGetTimerValue(cuProfile->kernelTime);
    cuProfile->kernelExecErr = cudaPeekAtLastError();
} 

extern  
void cuReduction(dim3 cuBlock, dim3 cuGrid, 
             Max* devMaxValues, int numRows,
             cudaProfile* cuProfile)
{
    cutCreateTimer(&(cuProfile->kernelTime));
    cutResetTimer(cuProfile->kernelTime);
    cutStartTimer(cuProfile->kernelTime);
    
    reduction<<<cuGrid, cuBlock>>> (devMaxValues, numRows); 
    cudaThreadSynchronize(); 
    
    cutStopTimer(cuProfile->kernelTime);
    
    cuProfile->kernelDuration = cutGetTimerValue(cuProfile->kernelTime);
    cuProfile->kernelExecErr = cudaPeekAtLastError();
} 

#endif // CU_EXTERNS_TEST_CU_

