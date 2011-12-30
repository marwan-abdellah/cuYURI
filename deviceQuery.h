#pragma once 

#include "cuShared.h"
#include "deviceQueryFile.h"
#include "util.h"

/* Shared Utilities "QA Testing" */
#include <shrUtils.h>
#include <shrQATest.h>
#include <memory>
#include <iostream>

/* CUDA-C */ 
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>

namespace deviceQuery
{
	/* */
	extern int deviceCount; 
	extern int driverVersion = 0; 
	extern int runtimeVersion = 0;
	
	/* @queryGPU */
	void queryGPU( int argc, char** argv) ; 
}


struct deviceInfo
{

}; 

deviceInfo* getDeviceInfo(int deviceNo, bool logToInfoFile)
{
	deviceInfo gpuInfo;


	return &gpuInfo; 
}