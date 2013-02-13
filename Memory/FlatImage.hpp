#include "FlatImage.h"
#include "Globals.h"
#include "MACROS/MACROS.h"

namespace Yuri
{
namespace Memory
{
// Constructor
template <class T>
FlatImage<T>::FlatImage
(const int NX, const int NY) : NX(NX), NY(NY)
{
    INFO("Creating a FlatImage object of size : " +
         CATS("[") + ITS(NX) + CATS("] X [") + ITS(NY) + CATS("]"));

    // Total number of pixels in the image
    this->NXY = NX * NY;

    // FlatImage size in Bytes
    this->imageSize = sizeof(T) * this->NXY;

    // Allocating the image data memory block
    FlatImage::AllocateImage();
}

template <class T>
FlatImage<T>::~FlatImage()
{
    INFO("Destructing the FlatImage object");

    // Freeing the image data
    FlatImage::FreeImage();

    INFO("The FlatImage object has been successfully destructed");
}

template <class T>
void FlatImage<T>::AllocateImage()
{
    INFO("Allocating a FLAT Image : " +
         ITS(this->NXY) + CATS(" pixels") + CATS(" - ") +
         ITS(this->imageSize) + CATS(" Bytes"));

    // Image allocation
    this->ptrVolumeData = (T*) malloc (this->imageSize);

    INFO("The FlatImage has been successfully allocated 3D array");
}

template <class T>
T* FlatImage<T>::getPtrImageData() const
{
    // Getting a pointer to the FlatImage data
    return this->ptrImageData;
}

template <class T>
void FlatImage<T>::FreeImage()
{
    INFO("Freeing a 2D image data block of size: " +
         CATS("[") + ITS(this->NX) + CATS("] X [") +
         ITS(this->NY) + CATS("]") + CATS(" - ") +
         ITS(this->imageSize) + " Bytes");

    // Freeing image data memory block
    free(this->ptrFlatImageData);

    // Nulling the dangling pointer
    this->ptrFlatImageData = NULL;

    INFO("Freeing 2D image data has been done successfully");
}

// Dimentions retrival NX
template <class T>
inline int FlatImage<T>::getNX() const { return this->NX; }

// Dimentions retrival NY
template <class T>
inline int FlatImage<T>::getNY() const { return this->NY; }

} // namespace Memory
} // namespace Yuri
