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
void Init()
{
    INFO("Initializing Data & compiling shaders");

    InitGLEW();

    // Initializing Cg context
    InitCgContext();

    // Enable culling
    glEnable(GL_CULL_FACE);

    // Clearing color buffer
    glClearColor(0.0, 0.0, 0.0, 0);

    // Creating volume texture and uploading it to the GPU
    CreateVolumeTexture();

    // Creating the frame buffer
    CreateFrameBuffer();

    // Creating the color map texture
    CreateColorMapTexture();

    // Creating the render buffer
    CreateRenderBuffer();
}

int VolRayCaster(int argc, char** argv)
{
    INFO("Volume Raycasting w/ OpenGL - cuYURI Integration");
    InitGlut(argc, argv);

    // Registering OpenGL callbacks
    RegisterOpenGLCallBacks();

    // Resizing OpenGL window
    Resize_GL(WINDOW_SIZE,WINDOW_SIZE);

    // Initializing Data & Shaders
    Init();

    // GLUT main loop
    glutMainLoop();

    return 0;
}
}
