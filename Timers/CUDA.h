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

#ifndef _CUDA_TIMERS_H_
#define _CUDA_TIMERS_H_

#include "Timers/TimerIncludes.h"
#include "CUDA/cuGlobals.h"

namespace Timers
{
namespace CUDA
{

/*!
 *
 */
cuTimer InitTimer();

/*!
 *
 */
durationStruct GetDuration(cuTimer cudaTimer);
}

}
#endif // _CUDA_TIMERS_H_
