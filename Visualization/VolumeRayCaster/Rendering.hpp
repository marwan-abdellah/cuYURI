

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

#ifndef _RENDERING_HPP_
#define _RENDERING_HPP_



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
void UpdateScene()
{
    UpdateVolume();
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0,
                    iWidth, iHeight, iDepth, GL_RGBA, GL_UNSIGNED_BYTE, rgbaImage);
    glutPostRedisplay();
}
void RenderFinalImage()
{
    // Disable depth testing
    glDisable(GL_DEPTH_TEST);

    // Texture the quad with the final reconstructed image
    glBegin(GL_QUADS);
        glTexCoord2f(0,0);      glVertex2f(0,0);
        glTexCoord2f(1,0);      glVertex2f(1,0);
        glTexCoord2f(1, 1);     glVertex2f(1, 1);
        glTexCoord2f(0, 1);     glVertex2f(0, 1);
    glEnd();

    // Get back the depth testing
    glEnable(GL_DEPTH_TEST);
}

// display the final image on the screen
void RenderBufferToScreen()
{
    // Clearing buffers
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // Loading I[]
    glLoadIdentity();

    // Enable texturing
    glEnable(GL_TEXTURE_2D);

    // Render the loaded volume or the color cube
    if(ToggleColorCube)
        glBindTexture(GL_TEXTURE_2D, imageTex_ID);
    else
        glBindTexture(GL_TEXTURE_2D, backfaceBuffer);

    // Reshaping the window
    ReshapeOrtho_GL(WINDOW_SIZE, WINDOW_SIZE);

    // Render the final image
    RenderFinalImage();

    // Disable texturing
    glDisable(GL_TEXTURE_2D);
}

// render the backface to the offscreen buffer backfaceBuffer
void RenderBackface()
{
    // Attach texture to buffer
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D, backfaceBuffer, 0);

    // Clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // Enable culling
    glEnable(GL_CULL_FACE);

    // Cukking the front face
    glCullFace(GL_FRONT);

    // Draw the color bounding cube
    DrawColoredCube(1.0,1.0, 1.0);

    // Disable culling
    glDisable(GL_CULL_FACE);
}

void RayCastingPass()
{
    // Attach texture to buffer
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D,
                              imageTex_ID, 0);

    // Clearing buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Enable shading progfiles
    cgGLEnableProfile(vertexProfile);
    cgGLEnableProfile(fragmentProfile);

    // Bind the shaders to the GPU for execution
    cgGLBindProgram(vertexProgram);
    cgGLBindProgram(fragmentProgram);

    // Update the shader parameters
    cgGLSetParameter1f(cgGetNamedParameter
                       (fragmentProgram, "samplingStep"), samplingStep);
    SetUnformTexParameter("tex", backfaceBuffer, fragmentProgram, cgParam_1);
    SetUnformTexParameter("volTexture", volumeTex_ID, fragmentProgram, cgVolumeTexture);

    // Enable culling
    glEnable(GL_CULL_FACE);

    // Cull the backface
    glCullFace(GL_BACK);

    // Draw the bounding colored cube
    DrawColoredCube(1.0,1.0, 1.0);

    // Disable the culling
    glDisable(GL_CULL_FACE);

    // Disable shader profiles
    cgGLDisableProfile(vertexProfile);
    cgGLDisableProfile(fragmentProfile);
}
}
#endif // _RENDERING_HPP_
