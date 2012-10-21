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

#ifndef _CU_KERNEL_CONF_H_
#define _CU_KERNEL_CONF_H_

#include "cuGlobals.h"

namespace cuUtils
{
/*! Automatically detects the GPU configuration and generates
 * optimized kernel configuration matching the size of the
 * 1D vector \param N
 *
 * @param N
 *          Length of the device vector.
 *
 * @param profile
 *          GPU profiling structure.
 *
 * @return Automatically calculated CUDA kernel configuration
 *
 */
kernelConf* AutoGenKernelConf_1D(const int N);
}


#endif // _CU_KERNEL_CONF_H_
