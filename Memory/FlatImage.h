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
