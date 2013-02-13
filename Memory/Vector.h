#ifndef VECTOR_H
#define VECTOR_H

namespace Yuri
{
namespace Memory
{

export
template <class T>
class Vector
{
public:
    // Constructor
    Vector(const int NX);

    // Destructor
    ~Vector();

    // Dimentions retrival functions
    int getNX() const;

    // Gets a pointer to the image data block in the memory
    T* getPtrVectorData() const;

private:
    // Volume dimensions
    const int NX;

    // Image size in Bytes
    int vectorSize;

    // Pointer to image data
    T* ptrVectorData;

    // Allocating image memory block
    void AllocateVector();

    // Freeing the volume memory block
    void FreeVector();
};

#include "Vector.hpp"

} // namespace Memory
} // namespace Yuri

#endif // VECTOR_H
