#include "configGPU.h"


gridConf* CUDA::GPU_CONFIG::config_1D(int numThreads_Total, int numThreads_PerBlock, bool POWER_OF_TWO_THREAD_NUM)
{
	gridConf config_GPU_GRID;
	int numBlocks_PerGrid; 

	if (POWER_OF_TWO_THREAD_NUM)
		numBlocks_PerGrid = numThreads_Total / numThreads_PerBlock;
	else 
		numBlocks_PerGrid = ((numThreads_Total + numThreads_PerBlock - 1) / numThreads_PerBlock);

	config_GPU_GRID.numThreadsTotal = numThreads_Total;
	config_GPU_GRID.numThreadsPerBlock = numThreads_PerBlock;  
	config_GPU_GRID.numBlockPerGrid = numBlocks_PerGrid; 

	return &config_GPU_GRID; 
}