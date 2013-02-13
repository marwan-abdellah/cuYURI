#include "Vector.h"
#include "Globals.h"
#include "MACROS/MACROS.h"

namespace Yuri
{
namespace Memory
{
// Constructor
template <class T>
Vector<T>::Vector
(const int NX) : NX(NX)
{
    INFO("Creating a Vector object of size : " +
         CATS("[") + ITS(NX) + CATS("]"));

    // Vector size in Bytes
    this->imageSize = sizeof(T) * this->NX;

    // Allocating the image data memory block
    Vector::AllocateVector();
}

template <class T>
Vector<T>::~Vector()
{
    INFO("Destructing the Vector object");

    // Freeing the image data
    Vector::FreeVector();

    INFO("The Vector object has been successfully destructed");
}

template <class T>
void Vector<T>::AllocateVector()
{
    INFO("Allocating a FLAT Image : " +
         ITS(this->NX) + CATS(" items") + CATS(" - ") +
         ITS(this->vectorSize) + CATS(" Bytes"));

    // Vector allocation
    this->ptrVectorData = (T*) malloc (this->vectorSize);

    INFO("The Vector has been successfully allocated");
}

template <class T>
T* Vector<T>::getPtrVectorData() const
{
    // Getting a pointer to the Vector data
    return this->ptrVectorData;
}

template <class T>
void Vector<T>::FreeVector()
{
    INFO("Freeing a vector data block of size: " +
         CATS("[") + ITS(this->NX) + CATS("]") + CATS(" - ") +
         ITS(this->vectorSize) + " Bytes");

    // Freeing image data memory block
    free(this->ptrVectorData);

    // Nulling the dangling pointer
    this->ptrVectorData = NULL;

    INFO("Freeing vector data has been done successfully");
}

// Dimentions retrival NX
template <class T>
inline int Vector<T>::getNX() const { return this->NX; }

} // namespace Memory
} // namespace Yuri
