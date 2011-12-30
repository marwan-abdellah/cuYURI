/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: bench.h
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#pragma once

#include "cuShared.h"
#include "app.h"
#include "util.h"
#include "timers.h"


namespace bench
{
	namespace cpu
	{
		void startTimer(stlTimer* _timer); 
		void stopTimer(stlTimer* _timer);  
	}

	namespace gpu
	{
		void startTimer(); 
		void stopTimer(); 
		void calcTime(); 	
	}
}

