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
