
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

#ifndef _GLUT_HPP_
#define _GLUT_HPP_


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




void InitGlut(int argc, char** argv)
{
    // Initialize GLUT context
    glutInit(&argc, argv);

    // Initial window dimensions
    const int initialWinWidth = 512;
    const int initialWinHeight = 512;

    // Calculate window dimensions and center its position
    const int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
    const int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
    const int glutWinPos_X = ((screenWidth - initialWinWidth) / 2);
    const int glutWinPos_Y = ((screenHeight- initialWinHeight) / 2);

    // RGBA and double buffering display mode
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

    // Set GLUT window size
    glutInitWindowSize(initialWinWidth, initialWinHeight);

    // Set GLUT window position
    glutInitWindowPosition(glutWinPos_X, glutWinPos_Y);

    // Create GLUT window and set its title
    glutCreateWindow("GPU Raycasting - cuYURI integration");

    // Reshape window size
    glutReshapeWindow(WINDOW_SIZE, WINDOW_SIZE);

    INFO("Display Mode : GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH");
    INFO("Initializing GLUT DONE");
}
}

#endif // _GLUT_HPP_
