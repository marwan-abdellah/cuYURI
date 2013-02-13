#include "FlatVolume.h"
#include "Globals.h"
#include "MACROS/MACROS.h"

// Constructor
export
template <class T>
FlatVolume<T>::FlatVolume
(const int NX, const int NY, const int NZ) : NX(NX), NY(NY), NZ(NZ)
{
    INFO("Creating a FLAT volume object of size: " +
         CATS("[") + ITS(NX) + CATS("] X [") +
         ITS(NY) + CATS("] X [") + ITS(NZ) + CATS("]"));

    // Total number of voxels in the volume
    NXYZ = NX * NY * NZ;

    volumeSize = sizeof(T) * NXYZ;

    FlatVolume::AllocateVolume();
}

export
template <class T>
FlatVolume<T>::~FlatVolume()
{
    INFO("Destructing the FlatVolume object");

    // Freeing the volume data
    FlatVolume::FreeVolume();

    INFO("The FlatVolume object has been successfully destructed");
}

export
template <class T>
void FlatVolume<T>::AllocateVolume()
{
    INFO("Allocating the volume :" + ITS(NXYZ) + CATS(" voxels"));
    INFO("Volume size : " + ITS(volumeSize) + CATS(" Bytes"));

    // Volume allocation
    T* ptrVolumeData = (T*) malloc (volumeSize);

    // Returning a pointer to the volume data
    this->ptrVolumeData = ptrVolumeData;

    INFO("The FLAT volume has been successfully allocated in 1D array");
}

export
template <class T>
T* FlatVolume<T>::getPtrVolumeData() const
{
    INFO("Getting a pointer to the volume data");

    return this->ptrVolumeData;
}

export
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

// Dimentions retrival NX
export
template <class T>
int FlatVolume<T>::getNX() const
{
    INFO("Fetching NX = " + ITS(this->NX));
    return this->NX;
}

// Dimentions retrival NY
export
template <class T>
int FlatVolume<T>::getNY() const
{
    INFO("Fetching NY = " + ITS(this->NY));
    return this->NY;
}

// Dimentions retrival NZ
export
template <class T>
int FlatVolume<T>::getNZ() const
{
    INFO("Fetching NZ = " + ITS(this->NZ));
    return this->NZ;
}


