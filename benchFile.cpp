/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: benchFile.cpp
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#include "benchFile.h"

namespace benchFile
{
	std::ofstream bfStream;

	void benchFile::createBenchFile(char* file);
	void benchFile::header(std::string cHeader);
	void benchFile::message(std::string cMessage);
	void benchFile::bench(stlTimer* cProfile, std::string cMessage);
	void benchFile::closeBenchFile(std::ofstream* benchFileStream); 
}; 

void benchFile::createBenchFile(char* file)
{
	/* Log file name */
	std::string fileName = (std::string)file;
	
	/* Log file extention ".bench" */
	std::string fileExtention = ".bench";

	/* @ For single hit string conversion to char array */
	std::string fileString = (std::string)("gen_") + fileName + fileExtention; 
	
	benchFile::bfStream.open((char*)fileString.c_str());

	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 
}

void benchFile::header(std::string cHeader)
{
	if (benchFile::bfStream != 0)
	{
		/* Getting time */
		time_t rawTime;
		struct tm* timeInfo;
		time(&rawTime); 
		timeInfo = localtime(&rawTime);

		benchFile::bfStream << "--------------------------------------------------" << std::endl;
		benchFile::bfStream << cHeader << std::endl;
	}
	else 
		std::cout << "Trace file stream is not available " << std::endl; 
}

void benchFile::message(std::string cMessage)
{
	/* Getting time */
	time_t rawTime;
	struct tm* timeInfo;
	time(&rawTime); 
	timeInfo = localtime(&rawTime); 

	benchFile::bfStream << "\t " << cMessage << std::endl;
}

void benchFile::bench(stlTimer* _timer, std::string cMessage)
{
	benchFile::bfStream << "--------------------------------------------------" << std::endl;
	benchFile::bfStream << "\t " << cMessage << std::endl;
	benchFile::bfStream << "\t @ sec(s)  " << _timer->get_seconds() << std::endl; 
	benchFile::bfStream << "\t @ msec(s) " << _timer->get_milliseconds() << std::endl; 
	benchFile::bfStream << "\t @ usec(s) " << _timer->get_microseconds() << std::endl; 
}

void benchFile::closeBenchFile(std::ofstream* benchFileStream)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	/* Closing the log file stream */
	benchFileStream->close();
}
