#ifndef _COPENGL_H_
#define _COPENGL_H_

#include "GL_Includes.h"


#define OFFSET(i) ((char *)NULL + (i))

#include <glut.h>
#include "GL/gl.h"
namespace OpenGL
{
// Initialization functions
void InitOpenGLContext(int argc, char** argv);
void InitOpenGL();



GLuint* UploadImageToTexture(const int imageWidth,
                             const int imageHeight,
                             float* imagePtr, GLuint* imageTex_ID);

GLuint* AllocateTex(const int imageWidth,
                                  const int imageHeight,
                                  float* imagePtr);

/* @ preparing FBO */
void prepareFBO(GLuint* iFBO_ID, GLuint* iSliceTexture_ID);
void updateSliceTexture(GLuint* iImageTexture_ID);

// OpenGL callbacks
void displayGL();
void reshapeGL(int iWinWidth, int iWinHeight);
void keyboardGL(unsigned char Button, int iX, int iY);
void idleGL();
void mouseGL(int iButton, int iState, int IX, int iY);
void mouseMotionGL(int iX, int iY);
void registerOpenGLCallBacks();
}

#endif // COPENGL_H
