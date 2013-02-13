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

#include "Image.h"
#include "Globals.h"
#include "MACROS/MACROS.h"

namespace Yuri
{
namespace Memory
{
// Constructor
template <class T> Image<T>::Image
(const int NX, const int NY) : NX(NX), NY(NY)
{
    INFO("Creating a 2D Image object of size : " +
         CATS("[") + ITS(NX) + CATS("] X [") + ITS(NY) + CATS("]"));

    // Total number of pixels in the image
    this->NXY = NX * NY;

    // Image size in Bytes
    ImageSize = sizeof(T) * this->NXY;

    // Allocating the Image data memory block
    Image::AllocateImage();
}

template <class T>
Image<T>::~Image()
{
    INFO("Destructing the Image object");

    // Freeing the Image data
    Image::FreeImage();

    INFO("The Image object has been successfully destructed");
}

template <class T>
void Image<T>::AllocateImage()
{
    INFO("Allocating a 2D Image : " +
         ITS(this->NXY) + CATS(" pixels") + CATS(" - ") +
         ITS(this->ImageSize) + CATS(" Bytes"));

    // 2D Image allocation
    this->ptrImageData = (T**) malloc (sizeof(T*) * this->NX);

    for(int j = 0; j < this->NX; j++)
    {
        this->ptrImageData[j] = (T*) malloc (sizeof(T) * this->NY);
    }

    INFO("The Image has been successfully allocated 3D array");
}

template <class T>
T** Image<T>::getPtrImageData() const
{
    // Getting a pointer to the Image data
    return this->ptrImageData;
}

template <class T>
void Image<T>::FreeImage()
{
    INFO("Freeing a 3D Image data block of size: " +
         CATS("[") + ITS(this->NX) + CATS("] X [") +
         ITS(this->NY) + CATS("]") + CATS(" - ") +
         ITS(this->ImageSize) + " Bytes");

    for(int j = 0; j < this->NY; j++)
    {
        // Release Y
        free(this->ptrImageData[j]);
    }

    // Release X
    free(this->ptrImageData);

    // Nulling the dangling pointer
    this->ptrImageData = NULL;

    INFO("Freeing 2D Image data has been done successfully");
}

// Dimentions retrival NX
template <class T>
inline int Image<T>::getNX() const { return this->NX; }

// Dimentions retrival NY
template <class T>
inline int Image<T>::getNY() const { return this->NY; }

} // namespace Memory
} // namespace Yuri
