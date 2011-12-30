/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: logFile.h
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#pragma once 

#include "cuShared.h"
#include "util.h"
#include "app.h"

namespace logFile
{
	/* */
	extern std::ofstream lfStream; 
	
	/* @createLogFile */
	void createLogFile(char* file); 
	
	/* @header */
	void header(std::string cHeader); 
	
	/* @message */
	void message(std::string cMessage); 
	
	/* @closeLogFile */
	void closeLogFile(std::ofstream* logFileStream); 
}


