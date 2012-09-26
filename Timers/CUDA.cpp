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

#include "Timers/CUDA.h"

cuTimer Timers::CUDA::InitTimer()
{
    cuTimer kernelTime;

	// Create the timer
	cutCreateTimer(&kernelTime);

	// Initialize the timer to ZEROs
	cutResetTimer(kernelTime);

	return kernelTime;
}

durationStruct Timers::CUDA::GetDuration(cuTimer cudaTimer)
{
    // Allocate durationStruct to get the execution time in
    // different reolutions
    durationStruct duration;

    // Get the execution time on GPU
    long _duration = cutGetTimerValue(cudaTimer);

    // Nano seconds
    duration.unit_NanoSec = _duration * 1000000;

    // Micro seconds
    duration.unit_MicroSec = _duration * 1000;

    // Milli seconds
    duration.unit_MilliSec = _duration;

    // Seconds
    duration.unit_Sec = _duration / 1000;

	return duration;
}



