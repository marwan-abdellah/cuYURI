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

#ifndef _GL_CALLBACKS_HPP_
#define _GL_CALLBACKS_HPP_

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
void Keyboard_GL(unsigned char fKey, int fX, int fY)
{
    // Dummy
    if (fX | fY | fKey) {}

    switch(fKey)
    {
    case 27:
        INFO("EXETING ... ");
        EXIT(0);
        break;
    case ' ':
        ToggleColorCube = !ToggleColorCube;
        INFO("Toggling the color cube and the volume");
        break;
    case 'x':
        rotationActive = !rotationActive;
        INFO("Auto rotation is active");
        break;

    case '[':
        samplingStep += 1.0 / 2048.0;
        INFO("Sampling step : " + FTS(samplingStep));
        break;
    case ']':
        samplingStep -= 1.0 / 2048.0;
        INFO("Sampling step : " + FTS(samplingStep));
        break;
    case 'q':
        rValueTF -= 0.05;
        INFO("Red : " + FTS(rValueTF));
    UpdateScene();
        break;
    case 'Q':
        rValueTF += 0.05;
        INFO("Red : " + FTS(rValueTF));
    UpdateScene();
        break;
    case 'w':
        gValueTF -= 0.05;
        INFO("Green : " + FTS(gValueTF));
    UpdateScene();
        break;
    case 'W':
        gValueTF += 0.05;
        INFO("Green : " + FTS(gValueTF));
    UpdateScene();
        break;
    case 'e':
        bValueTF -= 0.05;
        INFO("Blue : " + FTS(bValueTF));
    UpdateScene();
        break;
    case 'E':
        bValueTF += 0.05;
        INFO("Blue : " + FTS(bValueTF));
    UpdateScene();
        break;
    case 'r':
        aValueTF -= 0.05;
        INFO("Alpha : " + FTS(aValueTF));
    UpdateScene();
        break;
    case 'R':
        aValueTF += 0.05;
        INFO("Alpha : " + FTS(aValueTF));
    UpdateScene();
        break;
    case 't':
        desityThresholdTF -= 1;
        INFO("Threshold : " + FTS(desityThresholdTF));
    UpdateScene();
        break;
    case 'T':
        desityThresholdTF += 1;
        INFO("Threshold : " + FTS(desityThresholdTF));
    UpdateScene();
        break;
    case 'z':
        zoomValue *= 1.1;
        INFO("Zoom : " + FTS(zoomValue));
        break;
    case 'Z':
        zoomValue /= 1.1;
        INFO("Zoom : " + FTS(zoomValue));
        break;
    case 'a':
        rotateX += 5;
        INFO("rotateX : " + FTS(rotateX));
        break;
    case 'A':
        rotateX -= 5;
        INFO("rotateX : " + FTS(rotateX));
        break;
    case 's':
        rotateY += 5;
        INFO("rotateY : " + FTS(rotateY));
        break;
    case 'S':
        rotateY -= 5;
        INFO("rotateY : " + FTS(rotateY));
        break;
    case 'd':
        rotateZ += 5;
        INFO("rotateZ : " + FTS(rotateZ));
        break;
    case 'D':
        rotateZ -= 5;
        INFO("rotateZ : " + FTS(rotateZ));
        break;

    // Default case
    default:
        break;
    }

    glutPostRedisplay();
}

void Idel_GL()
{
    glutPostRedisplay();
}

void ReshapeOrtho_GL(int windowWidth, int windowHeight)
{
    if (windowHeight == 0)
        windowHeight = 1;

    // Adjust viewport
    glViewport(0, 0, windowWidth, windowHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Orthograhic projection
    gluOrtho2D(0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
}

void Resize_GL(int windowWidth, int windowHeight)
{
    if (windowHeight == 0)
        windowHeight = 1;

    // Adjust viewport
    glViewport(0, 0, windowWidth, windowHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Perspective projection
    gluPerspective(60.0, (GLfloat)windowWidth / (GLfloat)windowHeight, 0.01, 400.0);
    glMatrixMode(GL_MODELVIEW);
}

void Display_GL()
{
    if (rotationActive)
    {
        // Increment rotation
        rotateX += 0.25;
        rotateY += 0.25;
        rotateZ += 0.25;
    }

    // Resize window
    Resize_GL(WINDOW_SIZE,WINDOW_SIZE);

    // Enable render buffer
    EnableRenderBuffers();

    glLoadIdentity();

    // Z-axix translation
    glTranslatef(0, 0, -2.25);

    glScalef(zoomValue, zoomValue, zoomValue);

    // X-axis rotation
    glRotatef(rotateX, 1, 0, 0);

    // Y-axis rotation
    glRotatef(rotateY, 0, 1, 0);

    // Z-axis rotation
    glRotatef(rotateZ, 0, 0, 1);

    // Centring the cube
    glTranslatef(-0.5, -0.5, -0.5);

    // Render the backface of the cube
    RenderBackface();

    // Raycast the volume
    RayCastingPass();

    // Disable the render buffers
    DisableRenderBuffers();

    // Render the final image to the texture
    RenderBufferToScreen();

    // Swap buffers
    glutSwapBuffers();
}

void RegisterOpenGLCallBacks()
{
    // Keyboard function
    glutKeyboardFunc(Keyboard_GL);

    // Display function
    glutDisplayFunc(Display_GL);

    // Idle function
    glutIdleFunc(Idel_GL);

    // Reshaping function
    glutReshapeFunc(Resize_GL);
}
}

#endif // _GL_CALLBACKS_HPP_
