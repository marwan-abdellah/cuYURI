#ifndef FLATVOLUME_H
#define FLATVOLUME_H

#include <cstdlib>
#include <fftw3.h>
#include <cufft.h>
#include "Globals.h"
#include "MACROS/MACROS.h"

namespace Yuri
{
namespace Memory
{

export
template <class T>
class FlatVolume
{
public:
    // Constructor
    FlatVolume(const int NX, const int NY, const int NZ);

    // Destructor
    ~FlatVolume();

    // Dimentions retrival functions
    int getNX() const;
    int getNY() const;
    int getNZ() const;

    // Gets a pointer to the volume data block in the memory
    T* getPtrVolumeData() const;

private:
    // Volume dimensions
    const int NX;
    const int NY;
    const int NZ;
    int NXYZ;

    // Volume size in Bytes
    int volumeSize;

    // Pointer to volume data
    T* ptrVolumeData;

    // Allocating volume memory block
    void AllocateVolume();

    // Freeing the volume memory block
    void FreeVolume();
};

#include "FlatVolume.hpp"

} // namespace Memory
} // namespace Yuri


#endif // FLATVOLUME_H
