/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: sub_1D.cpp
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


namespace sub_1D
{
	extern char* appName; 
	
	/* @run_sub_1D */
	void run_sub_1D(int argc, char** argv); 
}