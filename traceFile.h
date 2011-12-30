/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: traceFile.h
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#pragma once

#include "cuShared.h"
#include "app.h"
#include "util.h"

namespace traceFile
{
	/* */
	extern std::ofstream tfStream; 
	
	/* @createTraceFile */
	void createTraceFile(char* file);  

	/* @header*/
	void header(std::string cMessage);
	
	/* @trace */
	void trace(std::string cMessage); 
	
	/* @closeTraceFile */
	void closeTraceFile(std::ofstream* traceFileStream); 
}