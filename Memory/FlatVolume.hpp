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

#include "FlatVolume.h"
#include "Globals.h"
#include "MACROS/MACROS.h"

// Constructor
template <class T> FlatVolume<T>::FlatVolume
(const int NX, const int NY, const int NZ) : NX(NX), NY(NY), NZ(NZ)
{
    INFO("Creating a FLAT volume object of size : " +
         CATS("[") + ITS(NX) + CATS("] X [") +
         ITS(NY) + CATS("] X [") + ITS(NZ) + CATS("]"));

    // Total number of voxels in the volume
    NXYZ = NX * NY * NZ;

    // Volume size in Bytes
    volumeSize = sizeof(T) * NXYZ;

    FlatVolume::AllocateVolume();
}

template <class T>
FlatVolume<T>::~FlatVolume()
{
    INFO("Destructing the FlatVolume object");

    // Freeing the volume data
    FlatVolume::FreeVolume();

    INFO("The FlatVolume object has been successfully destructed");
}

template <class T>
void FlatVolume<T>::AllocateVolume()
{
    INFO("Allocating the volume : " +
         ITS(NXYZ) + CATS(" voxels") + CATS(" - ") +
         ITS(volumeSize) + CATS(" Bytes"));

    // Volume allocation
    this->ptrVolumeData = (T*) malloc (volumeSize);

    INFO("The FLAT volume has been successfully allocated in 1D array");
}

template <class T>
T* FlatVolume<T>::getPtrVolumeData() const
{
    // Getting a pointer to the volume data
    return this->ptrVolumeData;
}

template <class T>
void FlatVolume<T>::FreeVolume()
{
    INFO("Freeing a volume data block of size: " +
         CATS("[") + ITS(NX) + CATS("] X [") +
         ITS(NY) + CATS("] X [") + ITS(NZ) + CATS("]") +
         CATS(" - ") + ITS(volumeSize) + " Bytes");

    // Release volume data memory block
    free(this->ptrVolumeData);

    INFO("Freeing volume data has been done successfully");
}

// Dimensions retrival NX
template <class T>
inline int FlatVolume<T>::getNX() const { return this->NX; }

// Dimensions retrival NY
template <class T>
inline int FlatVolume<T>::getNY() const { return this->NY; }

// Dimensions retrival NZ
template <class T>
inline int FlatVolume<T>::getNZ() const { return this->NZ; }


