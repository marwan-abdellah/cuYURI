/*********************************************************************
*
*
*
*
*
*
*
*
*
*********************************************************************/

#include "shared.h"

extern 
void cuAddArray_1D(const int* inArr_1, const int* inArr_2, int* outArr, int numItem, gridConf* configGPU);

/*********************************************************************
*
*
*
*********************************************************************/ 
__global__ 
void addArray_1D(const int* inArr_1, const int* inArr_2, int* outArr, int numItem)
{
    int _blockDim	= blockDim.x; 
    int _blockIdx	= blockIdx.x; 
    int _threadIdx	= threadIdx.x; 
    
    int _thread_ID;  
    _thread_ID = _blockDim * _blockIdx + _threadIdx;
    
    /*
    * Boundary condition  
    */  
    if (_thread_ID < numItem)
    {
        outArr[_thread_ID] = inArr_1[_thread_ID] + inArr_2[_thread_ID];
    }
}

void cuAddArray_1D(const int* inArr_1, const int* inArr_2, int* outArr, int numItem, gridConf* configGPU)
{
	addArray_1D <<< configGPU -> numBlockPerGrid, configGPU -> numThreadsPerBlock >>> (inArr_1, inArr_2, outArr, numItem); 
}

/*********************************************************************
*
*
*
*********************************************************************/
extern 
__global__ 
void cuSubArray_1D(const int* inArr_1, const int* inArr_2, int* outArr, int numItem)
{
    int _blockDim	= blockDim.x; 
    int _blockIdx	= blockIdx.x; 
    int _threadIdx	= threadIdx.x; 
    
    int _thread_ID;  
    _thread_ID = _blockDim * _blockIdx + _threadIdx;
    
    /*
    * Boundary condition  
    */  
    if (_thread_ID < numItem)
    {
        outArr[_thread_ID] = inArr_1[_thread_ID] - inArr_2[_thread_ID];
    }
}

/*********************************************************************
*
*
*
*********************************************************************/
extern 
__global__ 
void cuMulArray_1D(const int* inArr_1, const int* inArr_2, int* outArr, int numItem)
{
    int _blockDim	= blockDim.x; 
    int _blockIdx	= blockIdx.x; 
    int _threadIdx	= threadIdx.x; 
    
    int _thread_ID;  
    _thread_ID = _blockDim * _blockIdx + _threadIdx;
    
    /*
    * Boundary condition  
    */  
    if (_thread_ID < numItem)
    {
        outArr[_thread_ID] = inArr_1[_thread_ID] * inArr_2[_thread_ID];
    }
}

/*********************************************************************
*
*
*
*********************************************************************/
extern 
__global__ 
void cuDivArray_1D(const int* inArr_1, const int* inArr_2, int* outArr, int numItem)
{
    int _blockDim	= blockDim.x; 
    int _blockIdx	= blockIdx.x; 
    int _threadIdx	= threadIdx.x; 
    
    int _thread_ID;  
    _thread_ID = _blockDim * _blockIdx + _threadIdx;
    
    /*
    * Boundary condition  
    */  
    if (_thread_ID < numItem)
    {
        outArr[_thread_ID] = inArr_1[_thread_ID] / inArr_2[_thread_ID];
    }
}


