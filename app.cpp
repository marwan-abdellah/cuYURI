/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: app.cpp
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#include "app.h"

/* */
namespace app
{
	/* @definition */
	char* appName;

	void app::initApp(char* appName);
	void app::exitApp();
}

/* */
void app::initApp(char* appName)
{
	app::appName = appName; 

	logFile::createLogFile(app::appName); 
	traceFile::createTraceFile(app::appName);
	benchFile::createBenchFile(app::appName);

	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 
}

/* */
void app::exitApp()
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	benchFile::closeBenchFile(&benchFile::bfStream); 
	traceFile::closeTraceFile(&traceFile::tfStream); 
	logFile::closeLogFile(&logFile::lfStream);
}

