
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


#ifndef _GLEW_HPP_
#define _GLEW_HPP_

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



void InitGLEW()
{
    INFO("Initializing GLEW");
    GLenum glewError = glewInit();

    // Initialize all the OpenGL extensions
    glewGetExtension("glMultiTexCoord2fvARB");

    if(glewGetExtension("GL_EXT_frameBuffer_object"))
        INFO("GL_EXT_frameBuffer_object support");

    if(glewGetExtension("GL_EXT_renderBuffer_object"))
        INFO("GL_EXT_renderBuffer_object support");

    if(glewGetExtension("GL_ARB_vertex_buffer_object"))
        INFO("GL_ARB_vertex_buffer_object support");

    if(GL_ARB_multitexture)
        INFO("GL_ARB_multitexture support");

    if (glewGetExtension("GL_ARB_fragment_shader")      != GL_TRUE ||
        glewGetExtension("GL_ARB_vertex_shader")        != GL_TRUE ||
        glewGetExtension("GL_ARB_shader_objects")       != GL_TRUE ||
        glewGetExtension("GL_ARB_shading_language_100") != GL_TRUE)
    {
        INFO("Driver does not support OpenGL Shading Language");
        INFO("EXITING ... ");
        EXIT(0);
    }

    INFO("GLEW initialization DONE");
}
}
#endif // _GLEW_HPP_
