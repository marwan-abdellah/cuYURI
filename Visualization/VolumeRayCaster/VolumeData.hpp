

/*********************************************************************
 * Copyright © 2011-2012,
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

#ifndef _VOLUME_DATA_HPP_
#define _VOLUME_DATA_HPP_

#include "VolumeRayCaster.h"

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
    // Load volume (create sample volume)
    //volumeImage* volImagePtr = MEM_ALLOC_1D_GENERIC(volumeImage, 1);
    //volImagePtr = Volume::CreateTestVolume(256);

//    for (int i = 0; i < 1000; i++)
//        INFO(ITS(volImagePtr->volPtr[i]));
    //volImagePtr = Volume::LoadVolume("../Data/FOOT/FOOT");

    ReadVolume("../Data/CTData/CTData");
//    for (int i = 0; i < 10000; i++)
//            INFO(ITS(rgbaImage[i]));

    INFO("Uploading volume texture to the GPU");

    // Upload volume to a GPU texture
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glGenTextures(1, &volumeTex_ID);
    glBindTexture(GL_TEXTURE_3D, volumeTex_ID);
    // glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA,
                 256,
                 256,
                 256,
                 0, GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 rgbaImage);

    INFO("Volume texture created");
}
}


#endif // _VOLUME_DATA_HPP_
