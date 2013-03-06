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

#ifndef _PARAMETERS_H_
#define _PARAMETERS_H_


/*! \param OpenGL window size */
#define WINDOW_SIZE 512

/*! \param RGBA volume image */
GLubyte * rgbaImage;

/*! \param RGBA volume image */
GLubyte * luminanceImage;

/*! \param Bounding box flag */
int setBoundingBox = 0;

/*! \param Volume total voxels */
int numVoxels = 0;

/*! TF parameters */
float rValueTF = 1;
float gValueTF = 1;
float bValueTF = 1;
float aValueTF = 1;
float desityThresholdTF = 200;

/*! \param Bounding box flag */
bool ToggleColorCube = true;

/*! \param Cg context */
CGcontext cgContext;

/*! Vertex & Fragment profiles */
CGprofile vertexProfile;
CGprofile fragmentProfile;

/*! Shading parameters */
CGparameter cgParam_1;
CGparameter cgVolumeTexture;

/*! Shader programs */
CGprogram vertexProgram;
CGprogram fragmentProgram;

/*!  OpenGL buffers IDs */
GLuint renderBuffer;
GLuint frameBuffer;
GLuint backfaceBuffer;

/*! \param Cg context */
GLuint volumeTex_ID;

/*! \param Final image texture ID */
GLuint imageTex_ID;

/*! \param Sampling step for the volume */
float samplingStep = (1 / 50.0);

/*! Rotation parameters */
float rotateX = 0;
float rotateY = 0;
float rotateZ = 0;
float zoomValue = 1;

/*! Volume dimensiosn */
int iWidth;
int iHeight;
int iDepth;

/*! Automatic rotation */
int rotationActive = 1;

#endif // _PARAMETERS_H_
