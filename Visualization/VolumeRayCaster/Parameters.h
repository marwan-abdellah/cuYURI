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

#ifndef _PARAMETERS_H_
#define _PARAMETERS_H_

#include "VolumeRayCaster.h"

#define WINDOW_SIZE 512


GLubyte *image;
GLubyte * rgbaImage;
GLubyte * luminanceImage;
int setBoundingBox = 0;
int numVoxels = 0;

float rValueTF = 1;
float gValueTF = 1;
float bValueTF = 1;
float aValueTF = 1;
float desityThresholdTF = 200;

using namespace std;


bool ToggleColorCube = true;

// Cg context
CGcontext cgContext;

// Vertex & Fragment profiles
CGprofile vertexProfile;
CGprofile fragmentProfile;

// Shading parameters
CGparameter cgParam_1;
CGparameter cgVolumeTexture;

// Shader programs
CGprogram vertexProgram;
CGprogram fragmentProgram;

// OpenGL buffers IDs
GLuint renderBuffer;
GLuint frameBuffer;
GLuint backfaceBuffer;

// Volume texture ID
GLuint volumeTex_ID;

// Final image texture ID
GLuint imageTex_ID;

// Sampling step for the volume
float samplingStep = (1 / 50.0);

float rotateX = 0;
float rotateY = 0;
float rotateZ = 0;
float zoomValue = 1;

int iWidth;
int iHeight;
int iDepth;


#endif // _PARAMETERS_H_
