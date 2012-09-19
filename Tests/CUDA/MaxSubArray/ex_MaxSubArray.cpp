 /*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File         : ex_MaxSubArray.cpp
 * Author(s)    : Ahmed Abdel Salam <ahmed.abusalim.89@gmail.com>
		  Marwan Abdellah   <abdellah.marwan@gmail.com> 
		  Salah Saleh	    <salah_saleh@live.com>	  
 * Created      : July 2011
 * Description  :
 * Note(s)      :
 *********************************************************************/

#include "ex_MaxSubArray.h"
#include "FFT/FFTShift.h"
#include "CUDA/cu_Utilities.h"

#include "Utilities/Utils.h"
#include "CUDA/cuGlobals.h"
#include "CUDA/cuExterns.h"
#include "cuExternsTest.h"
#include "CUDA/Interfaces/Fill_1D_Array_RND.h"

using std::cout;
using std::endl;
using std::string;

namespace ex_MaxSubArray
{
	/* @ Profilers */
    cudaProfile* cuProfile;
    cu_Profile cuProfile2;
	durationStruct* cpuProfile;

    cudaProfile* cuTotalProfile;
	durationStruct* cpuTotalProfile;
}


void ex_MaxSubArray::runSample()

{
    int N = 16;
    int *devArray;
    int *hostArray;
    hostArray = (int*) malloc (sizeof(int) * N);

    cudaMalloc((void**)(&devArray), N * sizeof(int));

    dim3 cuBlock(4, 1, 1);
    dim3 cuGrid(N / cuBlock.x, 1, 1);


    cuUtils::Upload_1D_Array <int> (NULL, NULL, 10);

    cu_Profile profiles;

    cuYURI::cu_Fill_1D_Array_RND <int> (cuBlock, cuGrid, devArray, N, &profiles);
    // Downloading array
    cuUtils::download_1D_int(hostArray, devArray, 16);

    for (int i = 0; i < N; i++)
        INFO(ITS(hostArray[i]));
}

void ex_MaxSubArray::readFile(char* fileName, int* inputArray, int numRows, int numCols)
{
	INFO("Reading file - Starting");

	// Array indicies
	int xIdx = 0;
	int yIdx = 0;

	// Input stream
	std::ifstream inStream(fileName);

	if (inStream)
	{
		// Reading lineRow by lineRow
		std::string lineRow;

		// Initializing the Y index
		yIdx = 0;

		// Getting line by line
		while (std::getline(inStream, lineRow))
		{
			// Getting column by column
			std::stringstream split(lineRow);

			int inputVal;

			// Resetting the X index
			xIdx = 0;
			while (split >> inputVal)
			{
				// storing the input value from the file to the array
				inputArray[((yIdx * numRows) + xIdx)] = inputVal;

				// Incrementing the X idex
				xIdx++;
			}

			// Incrementing the y index
			yIdx++;
		}
	}

	// Closing the input stream
	INFO("Closing the input stream");
	inStream.close();

	INFO("Reading file - Done");
}

/*
 * This functions is divided in 2 stages. In STAGE_1 we search for the maximum
 * for each row equals to g. So if the 2D array is 1024x1024 we will have 1024
 * maximum. In STAGE_2, we perform a reduction process to get the maximum of  
 * these maximums.
 */
void ex_MaxSubArray::getMax_CPU(int* inputArray, int numCores, int numRows, int numCols, int numItr, Sheet* xlSheet)
{
	INFO("Starting CPU implementation : Iterations " + ITS(numItr));

	/* CPU timing parameters */
	time_boost start, end;

	// Allocating CPU profiler
	cpuProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);
	cpuTotalProfile = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

	if (xlSheet)
	{
		// Averaging Rows
		xlSheet->writeStr(13, (6), "Avg");
		xlSheet->writeStr(15, (5), "ns");
		xlSheet->writeStr(16, (5), "us");
		xlSheet->writeStr(17, (5), "ms");
		xlSheet->writeStr(18, (5), "s");

		// Averaging Headers
		xlSheet->writeStr(14, (6), "S1_CPU");
		xlSheet->writeStr(14, (7), "S1_GPU");
		xlSheet->writeStr(14, (8), "S2_CPU");
		xlSheet->writeStr(14, (9), "S2_GPU");
		xlSheet->writeStr(14, (10), "T_CPU");
		xlSheet->writeStr(14, (11), "T_GPU");

		// Rows
		xlSheet->writeStr(3, (0), "# Itr");
		xlSheet->writeStr(3, (0), "ns");
		xlSheet->writeStr(4, (0), "us");
		xlSheet->writeStr(5, (0), "ms");
		xlSheet->writeStr(6, (0), "s");
		xlSheet->writeStr(7, (0), "cuErrors");

		// Initializing average profilers
		cpuTotalProfile->unit_NanoSec = 0;
		cpuTotalProfile->unit_MicroSec = 0;
		cpuTotalProfile->unit_MilliSec = 0;
		cpuTotalProfile->unit_Sec = 0;

		// Iterate to average the results
		for (int itr = 0; itr < numItr; itr++)
		{
			// Headers
			xlSheet->writeNum(1, ((itr * 6) + 1), itr);
			xlSheet->writeStr(2, ((itr * 6) + 1), "S1_CPU");
			xlSheet->writeStr(2, ((itr * 6) + 3), "S2_CPU");
			xlSheet->writeStr(2, ((itr * 6) + 5), "T_CPU");

		   	/*
			* An array for holding the maximum values of all
			* possible combination
			*/
			Max* maxValues = (Max*)malloc(sizeof(Max)*numRows*numRows);
		   
		   	/*
		   	* Every element will hold the summation of all elements till 
			* it's position (operating on the original input array)
			*/ 
			int* prefixSum = (int*)malloc(sizeof(int)*numRows*numCols) ;
			
			/*
			 * Start of parallel region inStream which we are going
			 * to divide numRows on the number of threads, each thread
			 * will calculate the maximum of all possible combination
			 * and only store the maximum of them all inStream maxVal
			 */
			#pragma omp parallel num_threads(numCores)
			{				
			   	
			    	// This variable will hold the strip prefix sum. 
				int stripPrefixSum = 0;

				// This variable will hold the maximum summation found
				int candMaxSubArr = 0;

				// This variable will hold the starting column index
				int strCol_Idx = 0;
				
				// Will hold the maximum value found
				int minVal = 0;

				// Will hold the minimum value found
				int maxVal = 0; 

				// @ STAGE_1 "Starting"
				#pragma omp single				
				start = Timers::BoostTimers::getTime_MicroSecond();

				// Copy the first row as it is
				#pragma omp for schedule(dynamic)
				for(int iCtr = 0; iCtr < numCols; iCtr++) 
					prefixSum[iCtr] = inputArray[iCtr];

				// Vertical summation (every element holds the summation of all the elements till it's position)
				#pragma omp for schedule(dynamic)
				for(int yCtr = 0; yCtr < numCols; yCtr++)
					for (int iCtr = 1; iCtr < numRows; iCtr++)					
						prefixSum[iCtr*numRows+yCtr] = prefixSum[(iCtr-1)*numRows+yCtr] + 
												inputArray[iCtr*numRows+yCtr];

				// Horizontal summation. Now prefix sum is obtained
				#pragma omp for schedule(dynamic)				
				for (int iCtr = 0; iCtr < numRows; iCtr++)
					for(int yCtr = 1; yCtr < numCols; yCtr++)					
						prefixSum[iCtr*numRows+yCtr] = prefixSum[iCtr*numRows+(yCtr-1)] + 
												prefixSum[iCtr*numRows+yCtr];  

			  	/* 
				* In this part we will go through all possible combinations of the 2D array
				* and will reduce the 2D problem into 1D
				*/
				#pragma omp for schedule(dynamic)
				for(int strRow_Idx = 0; strRow_Idx < numRows; strRow_Idx++)
				{
				   /* Instead of having endRow_Idx start from strRow_Idx and have rows*(rows+1)/2
					* Combination, we will make endRow_Idx start from zero so that we have a unique 
					* indexing
					*/
					for(int endRow_Idx = 0; endRow_Idx < numRows; endRow_Idx++)
					{
						// Set the current maximum value to zero
						maxValues[strRow_Idx*numRows+endRow_Idx].val = 0;

						// Check whether we are working within the right region
						if (endRow_Idx >= strRow_Idx)
						{
							// Setting prefix sum to zero
							stripPrefixSum = 0;
							
							// Setting starting column index to zero
							strCol_Idx = 0;
				
							// Set minVal to zero
							minVal = 0;

							// Set maxVal to zero
							maxVal = 0;
							
							// Loop on all the elements within a row
							for(int endCol_Idx = 0; endCol_Idx < numCols; endCol_Idx++)
							{
								// Calculating strip prefix sum for first row     
								if (strRow_Idx == 0)
									stripPrefixSum = prefixSum[endRow_Idx*numRows+endCol_Idx];
								else
									// Calculating the strip prefix sum
									stripPrefixSum = prefixSum[endRow_Idx*numRows+endCol_Idx] - prefixSum[(strRow_Idx-1)*numRows+endCol_Idx];

								// Calculating the candidate to be maximum value
								candMaxSubArr = stripPrefixSum - minVal;

								// Check if we have a new maximum
								if(candMaxSubArr > maxVal)
								{ 
									// Copying the info 
									maxVal = candMaxSubArr;
									maxValues[strRow_Idx*numRows+endRow_Idx].val = candMaxSubArr;
									maxValues[strRow_Idx*numRows+endRow_Idx].x1 = strRow_Idx;
									maxValues[strRow_Idx*numRows+endRow_Idx].y1 = strCol_Idx;
									maxValues[strRow_Idx*numRows+endRow_Idx].x2 = endRow_Idx;
									maxValues[strRow_Idx*numRows+endRow_Idx].y2 = endCol_Idx;
								}

								// Check if we have a new minimum
								if(stripPrefixSum < minVal)
								{
									// Set stripPrefixSum to zero
									minVal = stripPrefixSum;//stripPrefixSum = 0;
									// Set the starting column index to end column index plus one
									strCol_Idx = endCol_Idx + 1;
								}
							}
						}
					}
				}
			}

			// @ STAGE_1 "Done"
			end = Timers::BoostTimers::getTime_MicroSecond();

			// Calculate the duration of STAGE_1
			cpuProfile = Timers::BoostTimers::getDuration(start, end);

			// Printing profile data
			xlSheet->writeNum(3, ((itr * 6) + 1), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(4, ((itr * 6) + 1), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(5, ((itr * 6) + 1), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(6, ((itr * 6) + 1), cpuProfile->unit_Sec);

			int selectedMaxVal = 0;
			int indexMaxValue=0;

			// @ STAGE_2 "Starting"
			start = Timers::BoostTimers::getTime_MicroSecond();

			// Search for the maximum inputVal inStream all maximum candidates
			for (int iCtr = 0;  iCtr < (numRows*numRows); iCtr++)
			{
				if (maxValues[iCtr].val > selectedMaxVal)
				{
					// Set selected maximum value to the current found maximum
					selectedMaxVal = maxValues[iCtr].val;
					// Keeping track of the index
					indexMaxValue = iCtr;
				}
			}
			
//cout<< maxValues[indexMaxValue].val<<endl;
			// @ STAGE_2 "Done"
			end = Timers::BoostTimers::getTime_MicroSecond();

			// Calculate the duration for STAGE_2
			cpuProfile = Timers::BoostTimers::getDuration(start, end);

			// Printing the results
			xlSheet->writeNum(8, ((itr * 6) + 3), maxValues[indexMaxValue].y1);
			xlSheet->writeNum(9, ((itr * 6) + 3), maxValues[indexMaxValue].x1);
			xlSheet->writeNum(10, ((itr * 6) + 3), maxValues[indexMaxValue].y2);
			xlSheet->writeNum(11, ((itr * 6) + 3), maxValues[indexMaxValue].x2);
			xlSheet->writeNum(12, ((itr * 6) + 3), maxValues[indexMaxValue].val);

			// Printing profile data
			xlSheet->writeNum(3, ((itr * 6) + 3), cpuProfile->unit_NanoSec);
			xlSheet->writeNum(4, ((itr * 6) + 3), cpuProfile->unit_MicroSec);
			xlSheet->writeNum(5, ((itr * 6) + 3), cpuProfile->unit_MilliSec);
			xlSheet->writeNum(6, ((itr * 6) + 3), cpuProfile->unit_Sec);

			cpuTotalProfile->unit_NanoSec += cpuProfile->unit_NanoSec;
			cpuTotalProfile->unit_MicroSec += cpuProfile->unit_MicroSec;
			cpuTotalProfile->unit_MilliSec += cpuProfile->unit_MilliSec;
			cpuTotalProfile->unit_Sec += cpuProfile->unit_Sec;
			
		}

		// Printing average profile data
		xlSheet->writeNum(15, 6, (int) (cpuTotalProfile->unit_NanoSec / numItr));
		xlSheet->writeNum(16, 6, (int)(cpuTotalProfile->unit_MicroSec / numItr));
		xlSheet->writeNum(17, 6, (int)(cpuTotalProfile->unit_MilliSec / numItr));
		xlSheet->writeNum(18, 6, (int)(cpuTotalProfile->unit_Sec / numItr));
		
	}
	else
	{
		INFO("No valid XL sheet was created. Exiting ... ");
		EXIT(0);
	}

	FREE_MEM_1D(cpuProfile);
	FREE_MEM_1D(cpuTotalProfile);
	INFO("CPU implementation - Done");
}

/*
 * This functions is divided in 2 stages. In STAGE_1 we search for the maximum
 * for each row equals to g. So if the 2D array is 1024x1024 we will have 1024
 * maximum. In STAGE_2, we perform a reduction process to get the maximum of  
 * these maximums.
 */
void ex_MaxSubArray::getMax_CUDA(int* hostInputArray, int numRows, int numCols, int numItr, Sheet* xlSheet)
{
	INFO("Starting CUDA implementation");

	// Allocating the CUDA profiler
	cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);

	if (xlSheet)
	{
		for (int itr = 0; itr < numItr; itr++)
		{
			// Headers
			xlSheet->writeStr(2, ((itr * 6) + 2), "S1_GPU");
			xlSheet->writeStr(2, ((itr * 6) + 4), "S2_GPU");
			xlSheet->writeStr(2, ((itr * 6) + 6), "T_GPU");

			// Profile strcutures
			cuProfile = MEM_ALLOC_1D_GENERIC(cudaProfile, 1);


			// Memory required for input & output arrays
			INFO("Calculating memory required");
			const int inputArraySize  = sizeof(int) * numRows * numCols;
			const int prefixSum       = sizeof(int) * numRows * numCols;
			const int outputArraySize = sizeof(Max) * numRows * numRows;

			// Input & output arrays on the device side
			int* devInputArray;
			int* devPrefixSum;
			Max* devMaxValues;

			// the variable that will hold the maximum of all possible combination on the host side
			Max* hostMaxValue = (Max*)malloc(sizeof(Max)*1);//outputArraySize);

			// Allocating the device arrays
			INFO("Allocating device arrays");
			cutilSafeCall(cudaMalloc((void**)&devInputArray, inputArraySize));
			cutilSafeCall(cudaMalloc((void**)&devPrefixSum, prefixSum));
			cutilSafeCall(cudaMalloc((void**)&devMaxValues, outputArraySize));

			// Upload the input array to the device side
			INFO("Uploading the input array to the GPU");
			cutilSafeCall(cudaMemcpy(devInputArray, hostInputArray, inputArraySize, cudaMemcpyHostToDevice));

			/////////////////////////////////////////////////////////////////////////////////////////////////

			// Configuring the GPU
			INFO("Addjusting Gridding configuration (cuPrefixSum)");
			dim3 cuBlock_1(128, 1, 1);
			dim3 cuGrid_1(numRows/cuBlock_1.x, 1, 1);


			// Invokig the CUDA kernel
			INFO("Invoking CUDA kernel (cuPrefixSum)");
			cuPrefixSum(cuBlock_1, cuGrid_1, devInputArray, devPrefixSum, numRows, numCols, cuProfile);

			// Printing profile data
			double tempcuPrefixSumDuration = cuProfile->kernelDuration;
			double tempcuPrefixSumExecErr =  cuProfile->kernelExecErr;

			// Checking if kernel execution failed or not
			cutilCheckMsg("Kernel (cuPrefixSum) execution failed \n");

			/////////////////////////////////////////////////////////////////////////////////////////////////

			// Configuring the GPU
			INFO("Addjusting Gridding configuration (cuGetMax)");
			dim3 cuBlock_2(16, 16, 1);
			dim3 cuGrid_2(numRows/cuBlock_2.x, numRows/cuBlock_2.y, 1);

			// Invokig the CUDA kernel
			INFO("Invoking CUDA kernel (cuGetMax)");
			cuGetMax(cuBlock_2, cuGrid_2, devMaxValues, devPrefixSum, numRows, numCols, cuProfile);

			// Printing profile data
			double tempcuGetMaxDuration = cuProfile->kernelDuration;
			double tempcuGetMaxSumExecErr =  cuProfile->kernelExecErr;

			// Checking if kernel execution failed or not
			// cutilCheckMsg("Kernel (cuGetMax) execution failed \n");
			
			////////////////////////////////////////////////////////////////////////////////////////////////////

			// Configuring the GPU
			INFO("Addjusting Gridding configuration (cuReduction)");
			dim3 cuBlock_3(512, 1, 1);
			dim3 cuGrid_3((numRows*numRows)/(2*cuBlock_3.x), 1, 1);


			// Invokig the CUDA kernel
			INFO("Invoking CUDA kernel (cuReduction)");
			cuReduction(cuBlock_3, cuGrid_3, devMaxValues, numRows, cuProfile);

			// Printing profile data
			xlSheet->writeNum(3, ((itr * 6) + 2), (cuProfile->kernelDuration + tempcuGetMaxDuration + tempcuPrefixSumDuration) 
																* 1000 * 1000);
			xlSheet->writeNum(4, ((itr * 6) + 2), (cuProfile->kernelDuration + tempcuGetMaxDuration + tempcuPrefixSumDuration) * 1000);
			xlSheet->writeNum(5, ((itr * 6) + 2), (cuProfile->kernelDuration + tempcuGetMaxDuration + tempcuPrefixSumDuration));
			xlSheet->writeNum(6, ((itr * 6) + 2), (cuProfile->kernelDuration + tempcuGetMaxDuration + tempcuPrefixSumDuration) / 1000);
			xlSheet->writeNum(7, ((itr * 6) + 2), (cuProfile->kernelExecErr + tempcuGetMaxSumExecErr + tempcuPrefixSumExecErr) );


			// Checking if kernel execution failed or not
			cutilCheckMsg("Kernel (cuReduction) execution failed \n");

			///////////////////////////////////////////////////////////////////////////////////////////////////////

			// Download the maxValues array to the host side
			INFO("Downloading the resulting array to the CPU");
			cutilSafeCall(cudaMemcpy(hostMaxValue, devMaxValues, sizeof(Max)*1, cudaMemcpyDeviceToHost));//outputArraySize

			// Freeing the allocated memory on the device
			INFO("Freeing the device memory");
			cudaFree(devMaxValues);
			
//			int selectedMaxVal = 0;
//			int indexMaxVal = 0;

//			// Search for the maximum value in all maximum candidates
//			for (int i = 0; i < (numRows*numRows); i++)
//			{
//				if (hostMaxValue[i].val > selectedMaxVal)
//				{
//					// Updating the selected values
//					selectedMaxVal = hostMaxValue[i].val;
//
//					// Updating the index
//					indexMaxVal = i;
//				}
//			}
//
			xlSheet->writeNum(8,  ((itr * 6) + 4), hostMaxValue->y1);//hostMaxValue[indexMaxVal].y1);
			xlSheet->writeNum(9,  ((itr * 6) + 4), hostMaxValue->x1);//hostMaxValue[indexMaxVal].x1);
			xlSheet->writeNum(10, ((itr * 6) + 4), hostMaxValue->y2);//hostMaxValue[indexMaxVal].y2);
			xlSheet->writeNum(11, ((itr * 6) + 4), hostMaxValue->x2);//hostMaxValue[indexMaxVal].x2);
			xlSheet->writeNum(12, ((itr * 6) + 4), hostMaxValue->val);//hostMaxValue[indexMaxVal].val);
			//xlSheet->writeNum(13, ((itr * 6) + 4), indexMaxVal);

			// Freeing the allocated memory on the host
			free(hostMaxValue);
		}
	}
	FREE_MEM_1D(cuProfile);
	INFO("CUDA implementation Done");
}

