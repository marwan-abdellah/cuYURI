#pragma once 

#include "cuShared.h"
#include "logFile.h"

/* Shared Utilities "QA Testing" */
#include <shrUtils.h>
#include <shrQATest.h>
#include <memory>
#include <iostream>

/* CUDA-C */ 
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>

namespace deviceQueryFile
{
	/* */
	extern std::ofstream dqfStream; 
	
	/* @createLogFile */
	void createInfoFile(char* file); 
	
	/* @header */
	void header(std::string cHeader); 
	
	/* @message */
	void message(std::string cMessage); 
	
	/* @closeLogFile */
	void closeInfoFile(std::ofstream* infoFileStream); 
}