 /*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File         : MaxSubArray.h
 * Author(s)    : Salah Saleh	    <salah_saleh@live.com>	  
 * Created      : July 2011
 * Description  :
 * Note(s)      :
 *********************************************************************/

#ifndef _MAXSUBARRAY_KERNEL_CU_
#define _MAXSUBARRAY_KERNEL_CU_

#include <cutil_inline.h>
#include "Shared.h"


__global__ 
void prefixSum(int numRows, int numCols, int* dev_inputArray, int* dev_prefixSum)
{
	// Calculating the correct strRow_Idx from the configuration  
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Copy the first row as it is
	dev_prefixSum[tid] = dev_inputArray[tid];
	
	__syncthreads();

	// Vertical summation (every element holds the summation of all the elements till it's position)
	for (int iCtr = 1; iCtr < numRows; iCtr++)	
		dev_prefixSum[iCtr*numCols+tid] = dev_prefixSum[(iCtr-1)*numCols+tid] + dev_inputArray[iCtr*numCols+tid];

	__syncthreads();

	// Horizontal summation. Now prefix sum is obtained
	for(int yCtr = 1; yCtr < numCols; yCtr++)					
		dev_prefixSum[tid*numRows+yCtr] = dev_prefixSum[tid*numRows+(yCtr-1)] + dev_prefixSum[tid*numRows+yCtr]; 

}


__global__ 
void findMax(int numRows, int numCols, Max* dev_maxValues, int* dev_prefixSum) 
{   
    	// Calculating the correct strRow_Idx from the configuration  
	int strRow_Idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Calculating the correct endRow_Idx from the configuration  
	int endRow_Idx = blockIdx.y * blockDim.y + threadIdx.y;

        /* 
	* This variable will hold horizontal summation of the elements of verticalSum. 
	* Note: this element will hold the prefix sum
	*/
	int stripPrefixSum = 0;

	// This variable will hold the maximum summation found
	int candMaxSubArr = 0;

	// This variable will hold the starting column index
	int strCol_Idx = 0;

	// Will hold the minimum value found
	int minVal = 0;

	// Will hold the maximum value found
	int maxVal = 0;

	// Set the current maximum value to zero
	dev_maxValues[strRow_Idx*numRows+endRow_Idx].val = 0;

	// Check whether we are working within the right region
	if (endRow_Idx >= strRow_Idx)
	{
							
		// Loop on all the elements within a row
		for(int endCol_Idx = 0; endCol_Idx < numCols; endCol_Idx++)
		{
			// Calculating strip prefix sum for first row
			if (strRow_Idx == 0)
				stripPrefixSum = dev_prefixSum[endRow_Idx*numRows+endCol_Idx];
			else
				// Calculating the strip prefix sum
				stripPrefixSum = dev_prefixSum[endRow_Idx*numRows+endCol_Idx] - 
											dev_prefixSum[(strRow_Idx-1)*numRows+endCol_Idx];

			// Calculating the candidate to be maximum value
			candMaxSubArr = stripPrefixSum - minVal;

			// Check if we have a new maximum
			if(candMaxSubArr > maxVal)
			{ 
				// Copying the info 
				maxVal = candMaxSubArr;
				dev_maxValues[strRow_Idx*numRows+endRow_Idx].val = candMaxSubArr;
				dev_maxValues[strRow_Idx*numRows+endRow_Idx].x1 = strRow_Idx;
				dev_maxValues[strRow_Idx*numRows+endRow_Idx].y1 = strCol_Idx;
				dev_maxValues[strRow_Idx*numRows+endRow_Idx].x2 = endRow_Idx;
				dev_maxValues[strRow_Idx*numRows+endRow_Idx].y2 = endCol_Idx;
			}
			
			// Check if we have a new minimum
			if(stripPrefixSum < minVal)
			{
				// Set strip prefix sum to zero
				minVal = stripPrefixSum;
				// Set the starting column index to end column index plus one
				strCol_Idx = endCol_Idx + 1;
			}
		}
	}


}




__global__ 
void reduction(Max* dev_maxValues, int numRows)
{

	// Calculating the correct thread index from the configuration 
	int tid = blockIdx.x * blockDim.x + threadIdx.x; 
	 
	// Working on the left half of the array to prevent divergence
	for(int i = numRows*numRows>>1; i > 0; i >>= 1) 
	{

	  	__syncthreads();

		// Check on thread index being in the left half of the array
		if(tid < i)
		{
			// Compare the left side element to the one in the right side
			if(dev_maxValues[tid+i].val > dev_maxValues[tid].val)
			{
				// Copy from right to left element
				dev_maxValues[tid].val = dev_maxValues[tid+i].val;
				dev_maxValues[tid].x1  = dev_maxValues[tid+i].x1;
				dev_maxValues[tid].y1  = dev_maxValues[tid+i].y1;	
				dev_maxValues[tid].x2  = dev_maxValues[tid+i].x2;
				dev_maxValues[tid].y2  = dev_maxValues[tid+i].y2;
			}
		}
	 }
	
}



#endif  //_MAXSUBARRAY_KERNEL_CU_

