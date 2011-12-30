/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: app.h
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#pragma once

#include "cuShared.h"

#include "logFile.h"
#include "traceFile.h"
#include "benchFile.h"


namespace app
{
	/* @declaration */
	extern char* appName; 

	/* @initApp */
	void initApp(char* appName);

	/* @exitApp */
	void exitApp(); 
}
