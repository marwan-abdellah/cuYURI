#include "globalMem_2D.h"

#define BLOCK_SIZE 16

/* Interface */
extern 
void add_2D_Array(float* inArr_1, 
				  float* inArr_2, 
				  float* outArr, 
				  int arrayWidth, 
				  int arrayHeight, 
				  pitchSize __pitch, 
				  dim3 __numBlocks, 
				  dim3 __numThreadsPerBlock);  

void run_2D_GLOBAL_MEMORY()
{
	int arrayWidth	= 4;
	int arrayHeight	= 4;

	bool SEQ = true; 

	/* Host allocation */
	float* inArr_1_H = (float*) malloc(arrayWidth * arrayHeight * sizeof(float));
	float* inArr_2_H = (float*) malloc(arrayWidth * arrayHeight * sizeof(float));
	float* outArr_H	= (float*) malloc(arrayWidth * arrayHeight * sizeof(float));

	/* Fill arrays */
	int index = 0; 
	if (SEQ)
	{
		int ctr = 0; 
		for(int j = 0; j < (arrayHeight); j++)
		{
			for(int i = 0; i < (arrayWidth); i++)
			{
				index = ((j * arrayWidth) + i);

				inArr_1_H[index] = (float) ctr++;  
				inArr_2_H[index] = (float) ctr++; 
				outArr_H[index] = (float) 0;
			}
		}
	}
	else 
	{
		for(int j = 0; j < (arrayHeight); j++)
		{
			for(int i = 0; i < (arrayWidth); i++)
			{
				index = ((j * arrayWidth) + i);

				inArr_1_H[index] =  (float)rand()/(float)RAND_MAX;
				inArr_2_H[index] = (float)rand()/(float)RAND_MAX;
				outArr_H[index] = 0; 
			}
		}
	}
	    
	/* Print host arrays */ 
	printf("inArr_1_H \n");
	print_2D_Array(inArr_1_H, arrayWidth, arrayHeight);
	printf("inArr_2_H \n");
	print_2D_Array(inArr_2_H, arrayWidth, arrayHeight);

	/* Device allocation + <__pitch> */
	float *inArr_1_D, *inArr_2_D, *outArr_D;
	size_t __pitch;
	cudaMallocPitch((void**)&inArr_1_D, &__pitch, arrayHeight * sizeof(float), arrayWidth);
	cudaMallocPitch((void**)&inArr_2_D, &__pitch, arrayHeight * sizeof(float), arrayWidth);
	cudaMallocPitch((void**)&outArr_D, &__pitch, arrayHeight * sizeof(float), arrayWidth);

	/* Print __pitch */
	printf("__pitch %d \n", (__pitch/sizeof(float))); 

	/* Uploading data */ 
	cudaMemcpy2D(inArr_1_D, __pitch, inArr_1_H, arrayHeight * sizeof(float), arrayHeight * sizeof(float), arrayWidth, cudaMemcpyHostToDevice);
	cudaMemcpy2D(inArr_2_D, __pitch, inArr_2_H, arrayHeight * sizeof(float), arrayHeight * sizeof(float), arrayWidth, cudaMemcpyHostToDevice);
 
	/* Gridding */
	dim3 __numBlocks(1,1,1);
	dim3 __numThreadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	__numBlocks.x = ((arrayWidth / BLOCK_SIZE) + (((arrayWidth) % BLOCK_SIZE) == 0 ? 0:1));
	__numBlocks.y = ((arrayHeight / BLOCK_SIZE) + (((arrayHeight) % BLOCK_SIZE) == 0 ? 0:1));

	/* Kernel invokation */
	add_2D_Array(inArr_1_D, inArr_2_D, outArr_D, arrayWidth, arrayHeight, __pitch, __numBlocks, __numThreadsPerBlock); 

	/* Synchronization */
	cudaThreadSynchronize();

	/* Download result */
	cudaMemcpy2D(outArr_H, arrayHeight * sizeof(float), outArr_D, __pitch, arrayHeight * sizeof(float), arrayWidth, cudaMemcpyDeviceToHost);

	/* Free device arrays */
	cudaFree(inArr_1_D);
	cudaFree(inArr_2_D);
	cudaFree(outArr_D);

	/* Display results */
	printf("outArr \n");
	print_2D_Array(outArr_H, arrayWidth, arrayHeight);
}

/* Print utility */
void print_2D_Array(float *in, int arrayWidth, int arrayHeight)
{
	for(int j = 0; j < arrayHeight; j++)
	{
		for(int i = 0; i < arrayWidth; i++)
		{
			printf("%f\t",in[(i * arrayHeight) + j]);
		}
		printf("\n");
	}
	printf("\n");
}