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

#include "Timers/Boost.h"
#include "MACROS/MACROS.h"

boostTimer Timers::Boost::GetTime_Second()
{
    return (boostTimer)
            boost::posix_time::second_clock::local_time();
}

boostTimer Timers::Boost::GetTime_MicroSecond()
{
    return (boostTimer)
            boost::posix_time::microsec_clock::local_time();
}

durationStruct* Timers::Boost::GetDuration
(boostTimer startTime, boostTimer endTime)
{
    // Allocate durationStruct to get the execution time in
    // different reolutions
    durationStruct* duration = MEM_ALLOC_1D_GENERIC(durationStruct, 1);

    // Calculate the execution time for a given CPU function
    boostDuration _duration = (endTime - startTime);

    // Nano seconds
    duration->unit_NanoSec = (double) _duration.total_nanoseconds();

    // Micro seconds
    duration->unit_MicroSec = (double) _duration.total_microseconds();

    // Milli seconds
    duration->unit_MilliSec = (double) _duration.total_microseconds() / 1000;

    // Seconds
    duration->unit_Sec = (double) _duration.total_microseconds()/ (1000 * 1000);

    return duration;
}

