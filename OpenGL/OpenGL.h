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

#ifndef _OPENGL_H_
#define _OPENGL_H_

#include "GL/gl.h"




namespace OpenGL
{

void InitOpenGLContext(int argc, char** argv);

void DisplayImage(float* imagePtr, int NX, int NY);

GLuint* UploadImageToTexture(const int imageWidth,
                             const int imageHeight,
                             float* imagePtr);

void UpdateTexID(GLuint* imgTex_ID);

void UpdateWindowParams(const int newWinWidth, const int newWinHeight);

// OpenGL callbacks
void DisplayGL();
void ReshapeGL(int iWinWidth, int iWinHeight);
void KeyboardGL(unsigned char Button, int iX, int iY);
void IdleGL();
void MouseGL(int iButton, int iState, int IX, int iY);
void MouseMotionGL(int iX, int iY);
void RegisterGLCallBacks();
}

#endif // _OPENGL_H_
