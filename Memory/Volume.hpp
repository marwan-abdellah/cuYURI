#include "Volume.h"
#include "Globals.h"
#include "MACROS/MACROS.h"

namespace Yuri
{
namespace Memory
{
// Constructor
template <class T> Volume<T>::Volume
(const int NX, const int NY, const int NZ) : NX(NX), NY(NY), NZ(NZ)
{
    INFO("Creating a 3D volume object of size : " +
         CATS("[") + ITS(NX) + CATS("] X [") +
         ITS(NY) + CATS("] X [") + ITS(NZ) + CATS("]"));

    // Total number of voxels in the volume
    this->NXYZ = NX * NY * NZ;

    // Volume size in Bytes
    volumeSize = sizeof(T) * NXYZ;

    // Allocating the volume data memory block
    Volume::AllocateVolume();
}

template <class T>
Volume<T>::~Volume()
{
    INFO("Destructing the Volume object");

    // Freeing the volume data
    Volume::FreeVolume();

    INFO("The Volume object has been successfully destructed");
}

template <class T>
void Volume<T>::AllocateVolume()
{
    INFO("Allocating a 3D volume : " +
         ITS(NXYZ) + CATS(" voxels") + CATS(" - ") +
         ITS(volumeSize) + CATS(" Bytes"));

    // 3D volume allocation
    this->ptrVolumeData = (T***) malloc (sizeof(T**) * this->NX);

    for(int i = 0; i < this->NX; i++)
    {
        this->ptrVolumeData[i] = (T**) malloc (sizeof(T*) * this->NY);
        {
            for (int j = 0; j < this->NY; j++)
            {
                this->ptrVolumeData[i][j] = (T*) malloc (sizeof(T) * this->NZ);
            }
        }
    }

    INFO("The volume has been successfully allocated 3D array");
}

template <class T>
T*** Volume<T>::getPtrVolumeData() const
{
    // Getting a pointer to the volume data
    return this->ptrVolumeData;
}

template <class T>
void Volume<T>::FreeVolume()
{
    INFO("Freeing a 3D volume data block of size: " +
         CATS("[") + ITS(this->NX) + CATS("] X [") +
         ITS(this->NY) + CATS("] X [") + ITS(this->NZ) + CATS("]") +
         CATS(" - ") + ITS(this->volumeSize) + " Bytes");

    for(int j = 0; j < this->NY; j++)
    {
        for(int k = 0; k < this->NZ; k++)
        {
            // Release Z
            free(this->ptrVolumeData[j][k]);
        }

        // Release Y
        free(this->ptrVolumeData[j]);
    }

    // Release X
    free(this->ptrVolumeData);

    // Nulling the dangling pointer
    this->ptrVolumeData = NULL;

    INFO("Freeing 3D volume data has been done successfully");
}

// Dimentions retrival NX
template <class T>
inline int Volume<T>::getNX() const { return this->NX; }

// Dimentions retrival NY
template <class T>
inline int Volume<T>::getNY() const { return this->NY; }

// Dimentions retrival NZ
template <class T>
inline int Volume<T>::getNZ() const { return this->NZ; }

} // namespace Memory
} // namespace Yuri
