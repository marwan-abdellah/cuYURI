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

#ifndef _COLORCUBE_HPP_
#define _COLORCUBE_HPP_

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

void CreateVertexWithColor(float x, float y, float z)
{
    // Generate vertex color RGB
    glColor3f(x, y, z);

    // Adjust texture corrdinate depnding on vertex color
    glMultiTexCoord3fARB(GL_TEXTURE1_ARB, x, y, z);

    // Draw vertex in position
    glVertex3f(x, y, z);
}

void DrawColoredCube(float x, float y, float z)
{
    glBegin(GL_QUADS);

        // Back side
        glNormal3f(0.0, 0.0, -1.0);
        CreateVertexWithColor(0.0, 0.0, 0.0);
        CreateVertexWithColor(0.0, y, 0.0);
        CreateVertexWithColor(x, y, 0.0);
        CreateVertexWithColor(x, 0.0, 0.0);

        // Front side
        glNormal3f(0.0, 0.0, 1.0);
        CreateVertexWithColor(0.0, 0.0, z);
        CreateVertexWithColor(x, 0.0, z);
        CreateVertexWithColor(x, y, z);
        CreateVertexWithColor(0.0, y, z);

        // Top side
        glNormal3f(0.0, 1.0, 0.0);
        CreateVertexWithColor(0.0, y, 0.0);
        CreateVertexWithColor(0.0, y, z);
        CreateVertexWithColor(x, y, z);
        CreateVertexWithColor(x, y, 0.0);

        // Bottom side
        glNormal3f(0.0, -1.0, 0.0);
        CreateVertexWithColor(0.0, 0.0, 0.0);
        CreateVertexWithColor(x, 0.0, 0.0);
        CreateVertexWithColor(x, 0.0, z);
        CreateVertexWithColor(0.0, 0.0, z);

        // Left side
        glNormal3f(-1.0, 0.0, 0.0);
        CreateVertexWithColor(0.0, 0.0, 0.0);
        CreateVertexWithColor(0.0, 0.0, z);
        CreateVertexWithColor(0.0, y, z);
        CreateVertexWithColor(0.0, y, 0.0);

        // Right side
        glNormal3f(1.0, 0.0, 0.0);
        CreateVertexWithColor(x, 0.0, 0.0);
        CreateVertexWithColor(x, y, 0.0);
        CreateVertexWithColor(x, y, z);
        CreateVertexWithColor(x, 0.0, z);

    glEnd();
}
}

#endif // _COLORCUBE_HPP_
