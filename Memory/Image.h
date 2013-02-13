#ifndef IMAGES_H
#define IMAGES_H

namespace Yuri
{
namespace Memory
{
export
template <class T>
class Image
{
public:
    // Constructor
    Image(const int NX, const int NY);

    // Destructor
    ~Image();

    // Dimentions retrival functions
    int getNX() const;
    int getNY() const;

    // Gets a pointer to the image data block in the memory
    T** getPtrImageData() const;

private:
    // Image dimensions
    const int NX;
    const int NY;
    int NXY;

    // Image size in Bytes
    int ImageSize;

    // Pointer to Image data
    T** ptrImageData;

    // Allocating Image memory block
    void AllocateImage();

    // Freeing the Image memory block
    void FreeImage();
};

#include "Image.hpp"

} // namespace Memory
} // namespace Yuri

#endif // IMAGES_H
