#include "cuTest.h"


/* Variables */ 
float* h_A;
float* h_B;
float* h_C;
float* d_A;
float* d_B;
float* d_C;
bool noprompt = false;

/* Prototypes */
void cleanupResources(void);
void randomInit(float*, int);
void parseArguments(int, char**);

/* GPU config */
gpuConfig cuConfig; 

extern
void add_kernel(float* in_1, float* in_2, float* out, int arraySize, gpuConfig* config_GPU);

/* @createArray */
float* createArray(int arrayLenght)
{
	float* ptrArray = (float*) malloc(sizeof(float) * arrayLenght); 
	return ptrArray; 
}

/* @createArray */
void uploadData(float* array_H, float* array_D, int arraySize)
{
	int dataSize = (arraySize * sizeof(float)); 
	cutilSafeCall(cudaMemcpy(array_D, array_H, dataSize, cudaMemcpyHostToDevice));
}

/* @downloadData */
void downloadData(float* array_H, float* array_D, int arraySize)
{
	int dataSize = (arraySize * sizeof(float)); 
	cutilSafeCall(cudaMemcpy(array_H, array_D, dataSize, cudaMemcpyDeviceToHost));
}

/* @createArray_GPU */
float* createArray_GPU(int arrayLenght)
{
	float* array_1D; 
	int dataSize = (arrayLenght * sizeof(float)); 
	cutilSafeCall(cudaMalloc((void**)&array_1D, dataSize));

	return array_1D; 
}

/* @parseArguments */
void parseArguments(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "--noprompt") == 0 || strcmp(argv[i], "-noprompt") == 0) 
        {
            noprompt = true;
            break;
        }
    }
}

/* @randomInit */
void randomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

/* @cleanupResources */
void cleanupResources(void)
{
    /* Free device memory */
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

    /* Free host memory */
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
        
    cutilDeviceReset();
}

/* cuTest "main" */
int cuTest(int argc, char** argv)
{
    shrQAStart(argc, argv);

    printf("Vector Addition\n");
    int N = 1024;
    size_t size = N * sizeof(float);
    parseArguments(argc, argv);
    
    /* Invoke kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    cuConfig.numThreads = N;
    cuConfig.numThreadsPerBlock = 256; 
    cuConfig.numBlocksPerGrid =  (N + threadsPerBlock - 1) / threadsPerBlock;
    
    /* Allocate input vectors h_A and h_B in host memory */
    h_A = createArray(N);
    h_B = createArray(N);
    h_C = createArray(N);

    /* Initialize input vectors */
    randomInit(h_A, N);
    randomInit(h_B, N);

    /* Allocate vectors in device memory */
    d_A = createArray_GPU(N);
    d_B = createArray_GPU(N);
    d_C = createArray_GPU(N);

    /* Copy vectors from host memory to device memory */ 
    uploadData(h_A, d_A, N);
    uploadData(h_B, d_B, N);

	add_kernel(d_A, d_B, d_C, N, &cuConfig); 

#ifdef _DEBUG
    cutilSafeCall( cutilDeviceSynchronize() );
#endif

    /* Copy result from device memory to host memory */
    downloadData(h_C, d_C, N);
    
    
    for (int i =0 ; i < N; i++)
		printf("@%d, %f \n", i,h_C[i] ); 
    
    /* Verify result */
    int i;
    for (i = 0; i < N; ++i) 
	{
        float sum = h_A[i] + h_B[i];
        if (fabs(h_C[i] - sum) > 1e-5)
            break;
    }

    cleanupResources();
    shrQAFinishExit(argc, (const char **)argv, (i==N) ? QA_PASSED : QA_FAILED);

	return 0; 
}



