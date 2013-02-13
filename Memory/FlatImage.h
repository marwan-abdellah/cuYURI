#ifndef FLATIMAGE_H
#define FLATIMAGE_H

namespace Yuri
{
namespace Memory
{

export
template <class T>
class FlatImage
{
public:
    // Constructor
    FlatImage(const int NX, const int NY);

    // Destructor
    ~FlatImage();

    // Dimentions retrival functions
    int getNX() const;
    int getNY() const;

    // Gets a pointer to the image data block in the memory
    T* getPtrImageData() const;

private:
    // Volume dimensions
    const int NX;
    const int NY;
    int NXY;

    // Image size in Bytes
    int imageSize;

    // Pointer to image data
    T* ptrImageData;

    // Allocating image memory block
    void AllocateImage();

    // Freeing the volume memory block
    void FreeImage();
};

#include "FlatImage.hpp"

} // namespace Memory
} // namespace Yuri

#endif // FLATIMAGE_H
