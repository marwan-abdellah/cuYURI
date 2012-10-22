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
#ifndef _EX_GPU_CONSTANT_OPERATIONS_H_
#define _EX_GPU_CONSTANT_OPERATIONS_H_

namespace ex_GPUConstantOperations
{
    void run(int argc, char* argv[]);
    void Add(int argc, char* argv[]);
    void Subtract(int argc, char* argv[]);
    void Multiply(int argc, char* argv[]);
    void Divide(int argc, char* argv[]);
}

#endif // _EX_GPU_CONSTANT_OPERATIONS_H_
