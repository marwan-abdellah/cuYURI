#pragma once 

#include "shared.h"

namespace CUDA
{
	namespace GPU_CONFIG
	{
		gridConf* config_1D(int numThreads_Total, int numThreads_PerBlock, bool POWER_OF_TWO_THREAD_NUM); 
	}
}