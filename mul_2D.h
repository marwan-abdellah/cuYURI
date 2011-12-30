/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: mul_2D.h
	Description		:
	Created			: 
	Note(s)			:
***************************************************************************
*/

#pragma once

#include "cuShared.h"

#include "arr_2D_FLAT.h"
#include "logFile.h"
#include "app.h"
#include "cu_arr_2D_GLOBAL.h"
#include "bench.h"


namespace mul_2D
{
	namespace globalMem
	{
		extern char* appName; 
		
		/* @run_add_1D */
		void run(int argc, char** argv);
	}
}