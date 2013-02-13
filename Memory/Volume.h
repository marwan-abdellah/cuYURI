/*********************************************************************
 * Copyright © 2011-2013,
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

#ifndef Volume_H
#define Volume_H

#include <cstdlib>
#include "Globals.h"
#include "MACROS/MACROS.h"

namespace Yuri
{
namespace Memory
{
export
template <class T>
class Volume
{
public:
    // Constructor
    Volume(const int NX, const int NY, const int NZ);

    // Destructor
    ~Volume();

    // Dimentions retrival functions
    int getNX() const;
    int getNY() const;
    int getNZ() const;

    // Gets a pointer to the volume data block in the memory
    T*** getPtrVolumeData() const;

private:
    // Volume dimensions
    const int NX;
    const int NY;
    const int NZ;
    int NXYZ;

    // Volume size in Bytes
    int volumeSize;

    // Pointer to volume data
    T*** ptrVolumeData;

    // Allocating volume memory block
    void AllocateVolume();

    // Freeing the volume memory block
    void FreeVolume();
};

#include "Volume.hpp"

} // namespace Memory
} // namespace Yuri

#endif // Volume_H
