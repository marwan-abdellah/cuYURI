/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: logFile.cpp
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#include "logFile.h"

namespace logFile
{
	std::ofstream lfStream;

	void logFile::createLogFile(char* file); 
	void logFile::header(std::string cHeader);
	void logFile::message(std::string cMessage);
	void logFile::closeLogFile(std::ofstream* logFileStream); 
}; 

void logFile::createLogFile(char* file)
{
	/* Log file name */
	std::string fileName = (std::string)file;
	
	/* Log file extention ".log" */
	std::string fileExtention = ".log";

	/* @ For single hit string conversion to char array */
	std::string fileString = (std::string)("gen_") +  fileName + fileExtention; 
	
	logFile::lfStream.open((char*)fileString.c_str());

	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 
}

void logFile::header(std::string cHeader)
{
	if (logFile::lfStream != 0)
	{
		/* Getting time */
		time_t rawTime;
		struct tm* timeInfo;
		time(&rawTime); 
		timeInfo = localtime(&rawTime);

		logFile::lfStream << "--------------------------------------------------" << std::endl;
		logFile::lfStream << cHeader << std::endl;
		logFile::lfStream << "@ " << asctime (timeInfo) << std::endl; 
	}
	else 
		std::cout << "Log file stream is not available " << std::endl; 
}

void logFile::message(std::string cMessage)
{
	/* Getting time */
	time_t rawTime;
	struct tm* timeInfo;
	time(&rawTime); 
	timeInfo = localtime(&rawTime); 

	logFile::lfStream << "\t " << cMessage << std::endl;
	logFile::lfStream << "\t @ " << asctime (timeInfo) << std::endl; 
}

void logFile::closeLogFile(std::ofstream* logFileStream)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	/* Closing the log file stream */
	logFileStream->close();
}
