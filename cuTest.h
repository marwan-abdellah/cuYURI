#pragma once

// Includes
#include <stdio.h>
#include <cutil_inline.h>
#include <shrQATest.h>

struct gpuConfig
{
	int numThreads; 
	int numThreadsPerBlock;
	int numBlocksPerGrid;
}; 