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

#ifndef RAYCASTER_H
#define RAYCASTER_H

#include <GL/glew.h>
#include <Cg/cg.h>
#include <Cg/cgGL.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cassert>
#include "Volume/Vector3.h"
#include "Volume/Volume.h"
#include "Volume/Loader.h"
#include "MACROS/MACROS.h"

#include "Parameters.h"

using namespace std;

namespace RayCaster
{
void UpdateScene();
void UpdateVolume();
void ReadVolume(char *prefix);
void ReadHeader(char *prefix, int &w, int &h, int &d);
void EnableRenderBuffers();
void DisableRenderBuffers();
void CreateVertexWithColor(float x, float y, float z);
void cgErrorCallback();
void SetUnformTexParameter(char* par, GLuint tex,const CGprogram &program,CGparameter param);
void LoadVertexProgram(CGprogram &v_program,char *shader_path, char *program_name);
void LoadFragmentProgram(CGprogram &f_program,char *shader_path, char *program_name);
void DrawColoredCube(float x, float y, float z);
void CreateVolumeTexture();
void Init();
void Keyboard_GL(unsigned char fKey, int fX, int fY);
void Idle_GL();
void ReshapeOrtho_GL(int w, int h);
void Resize_GL(int w, int h);
void RenderFinalImage();
void RenderBufferToScreen();
void RenderBackface();
void CreateFrameBuffer();
void CreateColorMapTexture();
void CreateRenderBuffer();
void InitGLEW();
void InitCgContext();
void RayCastingPass();
void Display_GL();
void InitGlut(int argc, char** argv);
int VolRayCaster(int argc, char** argv);
void RegisterOpenGLCallBacks();
}


#endif // VOLUMERAYCASTER_H
