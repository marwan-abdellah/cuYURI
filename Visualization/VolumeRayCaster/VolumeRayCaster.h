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

namespace RayCaster
{
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
void RayCastingPass();
void Display_GL();
int VolRayCaster(int argc, char* argv[]);
}
#endif // VOLUMERAYCASTER_H
