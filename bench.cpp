/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: bench.cpp
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#include "bench.h"

namespace bench
{
	namespace cpu
	{
		void startTimer(stlTimer* _timer); 
		void stopTimer(stlTimer* _timer);   
	}

	namespace gpu
	{

	}
}

void bench::cpu::startTimer(stlTimer* _timer)
{
	_timer -> start(); 
}

void bench::cpu::stopTimer(stlTimer* _timer)
{
	_timer -> stop(); 
}


