/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: cuShared.h
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#pragma once

#include <stdio.h>
#include <cutil_inline.h>
#include <shrQATest.h>
#include <fstream>
#include <strstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <typeinfo>
#include<cutil_inline.h>
#include <iomanip>
#include <limits>

struct gpuConfig_1D
{
	int numThreads; 
	int numThreadsPerBlock;
	int numBlocksPerGrid;
}; 

struct gpuConfig_XD
{
	dim3 numThreads; 
	dim3 numThreadsPerBlock;
	dim3 numBlocksPerGrid;
}; 

struct charRet
{
	char* _charPtr; 
	int _size; 
}; 


typedef size_t pitchSize; 