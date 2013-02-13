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
