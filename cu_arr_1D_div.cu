/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: cu_arr_1D_div.cu
	Description		:
	Created			: 
	Note(s)			:
***************************************************************************
*/

#ifndef _CU_1D_DIV_CU_
#define _CU_1D_DIV_CU_

/* */
__global__ 
void divArrays_kernel__i(const int* in_1, const int* in_2, int* out, int arraySize)
{
    int thd_ID;
    int blk_Dim = blockDim.x; 
    int blk_Idx = blockIdx.x; 
    int thd_Idx = threadIdx.x; 
    
    thd_ID = blk_Dim * blk_Idx + thd_Idx;
    
    if (thd_ID < arraySize)
        out[thd_ID] = in_1[thd_ID] / in_2[thd_ID];
}

/* */
__global__ 
void divArrays_kernel__f(const float* in_1, const float* in_2, float* out, int arraySize)
{
    int thd_ID;
    int blk_Dim = blockDim.x; 
    int blk_Idx = blockIdx.x; 
    int thd_Idx = threadIdx.x; 
    
    thd_ID = blk_Dim * blk_Idx + thd_Idx;
    
    if (thd_ID < arraySize)
        out[thd_ID] = in_1[thd_ID] / in_2[thd_ID];
}

/* */
__global__ 
void divArrays_kernel__d(const double* in_1, const double* in_2, double* out, int arraySize)
{
    int thd_ID;
    int blk_Dim = blockDim.x; 
    int blk_Idx = blockIdx.x; 
    int thd_Idx = threadIdx.x; 
    
    thd_ID = blk_Dim * blk_Idx + thd_Idx;
    
    if (thd_ID < arraySize)
        out[thd_ID] = in_1[thd_ID] / in_2[thd_ID];
}

#endif /* _CU_1D_DIV_CU_ */ 