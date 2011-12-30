/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: traceFile.cpp
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#include "traceFile.h"

namespace traceFile
{
	std::ofstream tfStream;

	void createTraceFile(char* file);
	void header(std::string cMessage);
	void trace(std::string cMessage);
	void closeTraceFile(std::ofstream* traceFileStream);
}; 

void traceFile::createTraceFile(char* file)
{
	/* Log file name */
	std::string fileName = util::charArrayToString(file);
	
	/* Log file extention ".trace" */
	std::string fileExtention = ".trace";

	/* @ For single hit string conversion to char array */
	std::string fileString = (std::string)("gen_") + fileName + fileExtention; 
	
	traceFile::tfStream.open((char*)fileString.c_str()); 
}

void traceFile::header(std::string cMessage)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	traceFile::tfStream << cMessage << std::endl;
}

void traceFile::trace(std::string cMessage)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	traceFile::tfStream << "\t" << cMessage; //  << std::endl;
}

void traceFile::closeTraceFile(std::ofstream* traceFileStream)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	/* Closing the log file stream */
	traceFileStream->close();
}

