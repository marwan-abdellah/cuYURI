/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: cu_arr_2D_mul_GM.cu
	Description		:
	Created			: 
	Note(s)			:
***************************************************************************
*/

#ifndef _CU_2D_MUL_GM_CU_
#define _CU_2D_MUL_GM_CU_

__global__ 
void mulArrays_kernel__i(int *inArr_1, int *inArr_2, int *outArr_2, size_t __pitch, int arrayWidth, int arrayHeight)
{
	/* Block Dim (2D) */
	int blockDim__x = blockDim.x; 
	int blockDim__y = blockDim.y;
	
	/* Block Index */
	int blockIdx__x = blockIdx.x; 
	int blockIdx__y = blockIdx.y;
	
	/* Thread Index (2D)*/ 
	int thdIdx__x = threadIdx.x; 
	int thdIdx__y = threadIdx.y; 
	
	/* Thread ID (2D) */
	const int thdId__x = blockDim__x * blockIdx__x + thdIdx__x;
	const int thdId__y = blockDim__y * blockIdx__y + thdIdx__y;

	/* Pitch Index */
	int pIndex = (__pitch / sizeof(int));
	
	/* 1D Index */
	int index = ((thdId__x * pIndex) + thdId__y); 
	
	if((thdId__x < arrayWidth) && (thdId__y < arrayHeight))
	{
		outArr_2[index] = inArr_1[index] * inArr_2[index];
	}
}

__global__ 
void mulArrays_kernel__f(float *inArr_1, float *inArr_2, float *outArr_2, size_t __pitch, int arrayWidth, int arrayHeight)
{
	/* Block Dim (2D) */
	int blockDim__x = blockDim.x; 
	int blockDim__y = blockDim.y;
	
	/* Block Index */
	int blockIdx__x = blockIdx.x; 
	int blockIdx__y = blockIdx.y;
	
	/* Thread Index (2D)*/ 
	int thdIdx__x = threadIdx.x; 
	int thdIdx__y = threadIdx.y; 
	
	/* Thread ID (2D) */
	const int thdId__x = blockDim__x * blockIdx__x + thdIdx__x;
	const int thdId__y = blockDim__y * blockIdx__y + thdIdx__y;

	/* Pitch Index */
	int pIndex = (__pitch / sizeof(float));
	
	/* 1D Index */
	int index = ((thdId__x * pIndex) + thdId__y); 
	
	if((thdId__x < arrayWidth) && (thdId__y < arrayHeight))
	{
		outArr_2[index] = inArr_1[index] * inArr_2[index];
	}
}

__global__ 
void mulArrays_kernel__d(double *inArr_1, double *inArr_2, double *outArr_2, size_t __pitch, int arrayWidth, int arrayHeight)
{
	/* Block Dim (2D) */
	int blockDim__x = blockDim.x; 
	int blockDim__y = blockDim.y;
	
	/* Block Index */
	int blockIdx__x = blockIdx.x; 
	int blockIdx__y = blockIdx.y;
	
	/* Thread Index (2D)*/ 
	int thdIdx__x = threadIdx.x; 
	int thdIdx__y = threadIdx.y; 
	
	/* Thread ID (2D) */
	const int thdId__x = blockDim__x * blockIdx__x + thdIdx__x;
	const int thdId__y = blockDim__y * blockIdx__y + thdIdx__y;

	/* Pitch Index */
	int pIndex = (__pitch / sizeof(double));
	
	/* 1D Index */
	int index = ((thdId__x * pIndex) + thdId__y); 
	
	if((thdId__x < arrayWidth) && (thdId__y < arrayHeight))
	{
		outArr_2[index] = inArr_1[index] * inArr_2[index];
	}
}

#endif /* _CU_2D_MUL_GM_CU_ */ 

