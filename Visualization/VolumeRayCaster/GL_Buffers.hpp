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

#ifndef _GL_BUFFERS_HPP_
#define _GL_BUFFERS_HPP_

/*!
 * @ Interfaces
 */
#include "VolumeRayCaster.h"
#include "Parameters.h"

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
void EnableRenderBuffers()
{
    // Binding the frame and render buffers
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, frameBuffer);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderBuffer);
}

void DisableRenderBuffers()
{
    // Unbinding the frame buffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void CreateFrameBuffer()
{
    // Create frame buffer and attache 2D texture to it
    glGenFramebuffersEXT(1, &frameBuffer);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,frameBuffer);

    glGenTextures(1, &backfaceBuffer);
    glBindTexture(GL_TEXTURE_2D, backfaceBuffer);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA16F_ARB, WINDOW_SIZE, WINDOW_SIZE,
                 0, GL_RGBA, GL_FLOAT, NULL);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D, backfaceBuffer, 0);
}

void CreateColorMapTexture()
{
    // Creating color map texture
    glGenTextures(1, &imageTex_ID);
    glBindTexture(GL_TEXTURE_2D, imageTex_ID);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA16F_ARB, WINDOW_SIZE, WINDOW_SIZE,
                 0, GL_RGBA, GL_FLOAT, NULL);
}

void CreateRenderBuffer()
{
    // Creating render buffer
    glGenRenderbuffersEXT(1, &renderBuffer);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderBuffer);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT,
                             WINDOW_SIZE, WINDOW_SIZE);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                                 GL_RENDERBUFFER_EXT, renderBuffer);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}
}
#endif // _GL_BUFFERS_HPP_
