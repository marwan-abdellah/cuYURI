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

#include "GLUT.h"
#include "MACROS/MACROS.h"

void OpenGL::InitGlut(int argc, char** argv)
{
    INFO("Initializing GLUT");

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
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

    // Set GLUT window size
    glutInitWindowSize(initialWinWidth, initialWinHeight);

    // Set GLUT window position
    glutInitWindowPosition(glutWinPos_X, glutWinPos_Y);

    // Create GLUT window and set its title
    glutCreateWindow("cuYURI - OpenGL Integration");

    INFO("Display Mode : GLUT_RGBA | GLUT_DOUBLE");
    INFO("Initializing GLUT DONE");
}
