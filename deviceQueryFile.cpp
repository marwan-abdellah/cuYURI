#include "deviceQueryFile.h"

/*
 * No need to log this functions in this module for the time being. 
 */
namespace deviceQueryFile
{
	std::ofstream dqfStream;

	void createInfoFile(char* file); 
	void header(std::string cHeader);
	void message(std::string cMessage);
	void closeInfoFile(std::ofstream* deviceQueryFileStream); 
}; 

void deviceQueryFile::createInfoFile(char* file)
{
	/* Log file name */
	std::string fileName = (std::string)file;
	
	/* Log file extention ".log" */
	std::string fileExtention = ".info";

	/* @ For single hit string conversion to char array */
	std::string fileString = (std::string)("GPU") + fileExtention; 
	
	deviceQueryFile::dqfStream.open((char*)fileString.c_str());
}

void deviceQueryFile::header(std::string cHeader)
{
	if (deviceQueryFile::dqfStream != 0)
	{
		/* Getting time */
		time_t rawTime;
		struct tm* timeInfo;
		time(&rawTime); 
		timeInfo = localtime(&rawTime);

		deviceQueryFile::dqfStream << "--------------------------------------------------" << std::endl;
		deviceQueryFile::dqfStream << cHeader << std::endl; 
	}
	else 
		std::cout << "GPU.info file stream is not available " << std::endl; 
}

void deviceQueryFile::message(std::string cMessage)
{
	/* Getting time */
	time_t rawTime;
	struct tm* timeInfo;
	time(&rawTime); 
	timeInfo = localtime(&rawTime); 

	deviceQueryFile::dqfStream << "\t " << cMessage << std::endl; 
}

void deviceQueryFile::closeInfoFile(std::ofstream* deviceQueryFileStream)
{
	/* Closing the log file stream */
	deviceQueryFileStream->close();
}
