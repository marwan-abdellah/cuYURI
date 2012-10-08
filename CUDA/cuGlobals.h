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

#ifndef _CU_GLOBALS_H_
#define _CU_GLOBALS_H_

#include "cuIncludes.h"

struct gpuProfileStruct
{
    unsigned int kernelTime;
	float kernelDuration;
	int kernelExecErr;
};

typedef gpuProfileStruct cuProfile;

enum YURI_RETURN
{
    SUCCESS = 1,
    FAIL = 0
};

struct kernelConf
{
    dim3 cuGrid;
    dim3 cuBlock;
};

#endif // _CU_GLOBALS_H_ 
