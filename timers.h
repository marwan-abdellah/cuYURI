/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: timers.h
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#pragma once 

#include "cuShared.h"

/* Time structs */
struct timeStruct
{
	double _sec; 
	double _millisec;
	double _microsec;
}; 

typedef timeStruct profile;

/* STLSoft Timers */ 
#include <platformstl/performance/performance_counter.hpp>

typedef platformstl::performance_counter stlTimer;
