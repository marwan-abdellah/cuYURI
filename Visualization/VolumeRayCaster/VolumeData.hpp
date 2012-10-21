/*********************************************************************
 * Copyright © 2007-2012,
 * Marwan Abdellah: <abdellah.marwan@gmail.com>
 *
 * This code is part of the Ray Casting Tutorial provided by
 * Peter Trier <trier@daimi.au.dk>

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

#ifndef _VOLUME_DATA_HPP_
#define _VOLUME_DATA_HPP_

/*!
 * @ Interfaces
 */
#include "VolumeRayCaster.h"

/*!
 * @ Implementations
 */
#include "Cg.hpp"
#include "ColorCube.hpp"
#include "GL_CallBacks.hpp"
#include "GL_Buffers.hpp"
#include "GLEW.hpp"
#include "GLUT.hpp"
#include "LoadingVolume.hpp"
#include "Rendering.hpp"
#include "VolumeData.hpp"

namespace RayCaster
{
void CreateVolumeTexture()
{
    // Load volume
    ReadVolume("../Data/CTData/CTData");


    INFO("Uploading volume texture to the GPU");

    // Pixel storages
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);

    // Generate the 3D texture and bind it
    glGenTextures(1, &volumeTex_ID);
    glBindTexture(GL_TEXTURE_3D, volumeTex_ID);

    // Adjusting texture parameters
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

    // Upload the initial volume to the GPU
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, iWidth, iHeight, iDepth,
                 0, GL_RGBA, GL_UNSIGNED_BYTE, rgbaImage);

    INFO("Volume texture created and uploaded to the GPU ");
}
}


#endif // _VOLUME_DATA_HPP_
