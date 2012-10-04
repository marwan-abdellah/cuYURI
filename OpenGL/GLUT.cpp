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
#include "Utilities/MACROS.h"
void OpenGL::InitGlut(int argc, char** argv)
{
    INFO("Initializing GLUT");
    glutInit(&argc, argv);

    const int initialWinWidth = 512;
    const int initialWinHeight = 512;

    // Calculte window position
    const int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
    const int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
    const int glutWinPos_X = ((screenWidth - initialWinWidth) / 2);
    const int glutWinPos_Y = ((screenHeight- initialWinHeight) / 2);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(initialWinWidth, initialWinHeight);
    glutInitWindowPosition(glutWinPos_X, glutWinPos_Y);
    glutCreateWindow("cuYURI - OpenGL Integration");

    INFO("Display Mode : GLUT_RGBA | GLUT_DOUBLE");
    INFO("Initializing GLUT DONE");
}
