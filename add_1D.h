/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: add_1D.h
	Description		:
	Created			: 
	Note(s)			:
***************************************************************************
*/

#pragma once

#include "cuShared.h"

#include "arr_1D.h"
#include "logFile.h"
#include "app.h"
#include "cu_arr_1D.h"
#include "bench.h"


namespace add_1D
{
	extern char* appName; 
	
	/* @run_add_1D */
	void run_add_1D(int argc, char** argv); 
}