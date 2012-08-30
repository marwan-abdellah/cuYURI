 /*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File         : cuExternsTest.h
 * Author(s)    : Marwan Abdellah   <abdellah.marwan@gmail.com> 
		  Salah Saleh	    <salah_saleh@live.com>	  
 * Created      : July 2011
 * Description  :
 * Note(s)      :
 *********************************************************************/

#ifndef CUEXTERNS_TEST_H_
#define CUEXTERNS_TEST_H_

#include "CUDA/cuGlobals.h"

extern  
void cuPrefixSum(dim3 cuBlock, dim3 cuGrid, int* devInputArray, int* devPrefixSum, 
              int numRows, int numCols, cudaProfile* cuProfile);

extern  
void cuGetMax(dim3 cuBlock, dim3 cuGrid, Max* devMaxValues, int* devPrefixSum, 
              int numRows, int numCols, cudaProfile* cuProfile);

extern
void cuReduction(dim3 cuBlock, dim3 cuGrid, Max* devMaxValues, int numRows,
		cudaProfile* cuProfile);

#endif /* CUEXTERNS_TEST_H_ */

