/*********************************************************************
 * Copyright © 2011-2012,
 * Marwan Abdellah: <abdellah.marwan@gmail.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation.

 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.

 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 ********************************************************************/

#ifndef _TIMER_GLOBALS_H_
#define _TIMER_GLOBALS_H_

// includes, Boost timers
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>

// typedefs, Boost timer
typedef boost::posix_time::ptime boostTimer;
typedef boost::posix_time::time_duration boostDuration;

// typedef, CUDA timer
typedef uint cuTimer;

// Duration structure for different time resolutions
struct durationStruct
{
	double unit_NanoSec;
	double unit_MicroSec;
	double unit_MilliSec;
	double unit_Sec;
};


#endif // _TIMER_GLOBALS_H_
