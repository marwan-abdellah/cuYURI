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

#include "Globals.h"
#include "MACROS/MACROS.h"
#include "CUDA/Utilities/cuUtilities.h"

void cuUtils::DisplayKernelProfilingData(char* kernelName, cuProfile* profile)
{
    COUT << TAB; SEP();
    COUT << TAB << "Kernel Execution Time @[" << CATS(kernelName)  << "]" << ENDL;
    COUT << TAB << TAB << "@Nano-Seconds (ns)  : " << (profile->kernelDuration * (1000 * 1000)) << ENDL;
    COUT << TAB << TAB << "@Micro-Seconds (us) : " << (profile->kernelDuration * 1000) << ENDL;
    COUT << TAB << TAB << "@Milli-Seconds (ms) : " <<  profile->kernelDuration << ENDL;
    COUT << TAB << TAB << "@Seconds (s)        : " << (profile->kernelDuration / (1000)) << ENDL;
    COUT << TAB; SEP();
}


