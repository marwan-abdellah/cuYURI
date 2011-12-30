/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: div_1D.h
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


namespace div_1D
{
	extern char* appName; 
	
	/* @run_div_1D */
	void run_div_1D(int argc, char** argv); 
}