/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: benchFile.h
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#pragma once 

#include "cuShared.h"
#include "util.h"
#include "app.h"
#include "timers.h"
#include "bench.h"

namespace benchFile
{
	/* */
	extern std::ofstream bfStream; 
	
	/* @createBenchFile */
	void createBenchFile(char* file); 
	
	/* @header */
	void header(std::string cHeader); 
	
	/* @message */
	void message(std::string cMessage); 

	/* @profile */
	void bench(stlTimer* cProfile, std::string cMessage);
	
	/* @closeBenchFile */
	void closeBenchFile(std::ofstream* logFileStream); 
}

