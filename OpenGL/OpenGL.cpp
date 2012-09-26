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

#include "OpenGL.h"
#include "OpenGL/GLUT.h"
#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"


/************/
/* @ LOCALS */
/************/
GLuint* cGL_ImageTexture_ID;

int    eWinWidth;
int    eWinHeight;


int    glWinWidth       = 512;
int    glWinHeight      = 512;
float  glZoomLevel      = 1;


void OpenGL::UpdateWindowParams(const int newWinWidth, const int newWinHeight)
{
    glWinWidth = newWinWidth;
    glWinHeight = newWinHeight;

}

void OpenGL::InitOpenGLContext(int argc, char** argv)
{
    INFO ("Initializing OpenGL Contex");

    /* @ GLUT Initialization */
    OpenGL::InitGlut(argc, argv);


    /*
    // Initialize necessary OpenGL extensions
    if (!checkGLExtensions())
    {
        INFO("Missing OpenGL Necessary Extensions");
        EXIT(0);
    }
    else
        INFO("Requied OpenGL extensions are FOUND");
        */

    /* @ Clearing color buffer */
    glClearColor (0.0, 0.0, 0.0, 0.0);

    /* Setting the pixel storage mode */
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    INFO("Registering OpenGL callbacks");

    // Registering OpenGL CallBack Functions
    RegisterGLCallBacks();

    INFO("Initializing OpenGL Contex DONE");
}



GLuint* OpenGL::UploadImageToTexture(const int imageWidth,
                                  const int imageHeight,
                                  float* imagePtr)
{
    // Texture ID
    GLuint* imageTex_ID = MEM_ALLOC_1D_GENERIC(GLuint, 1);

    //  Create 2D texture object as a render target
    glGenTextures(1, imageTex_ID);

    // Bind the texture target
    glBindTexture(GL_TEXTURE_2D, *imageTex_ID);


    // Set the 2D texture parameters
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    // @NOTE Automatic mipmap Generation included in OpenGL v1.4
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                 imageWidth, imageHeight, 0,
                 GL_LUMINANCE, GL_FLOAT, imagePtr);

    // Unbinding texture
    glBindTexture(GL_TEXTURE_2D, 0);

    return imageTex_ID;
}

void OpenGL::UpdateTexID(GLuint* imgTex_ID)
{
    cGL_ImageTexture_ID = imgTex_ID;
}


void OpenGL::DisplayImage(float* imagePtr, int NX, int NY)
{
    OpenGL::InitOpenGLContext(NULL, NULL);

    GLuint* imgTex_ID = OpenGL::UploadImageToTexture(NX, NY, imagePtr);

    OpenGL::UpdateTexID(imgTex_ID);

    glutMainLoop();
}



void OpenGL::DisplayGL()
{
    /* @ Clearing color buffer */
    glClear(GL_COLOR_BUFFER_BIT);

    /* @ Disabling depth test */
    glDisable(GL_DEPTH_TEST);

    /* @ Binding slice texture to be displayed On OpenGL Quad */
    glBindTexture(GL_TEXTURE_2D, *cGL_ImageTexture_ID);
    glEnable(GL_TEXTURE_2D);

    /* Adjusting slice texture parameters */
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    /* @ Adjusting viewport */
    glViewport(-glWinWidth / 2, -glWinHeight / 2, glWinWidth * 2, glWinHeight * 2);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    /* @ Center slice texture at the orgin (0,0) */
    glScalef(glZoomLevel, glZoomLevel, 1);
    glTranslatef(-0.5, -0.5, 0.0);

    /* @ Texture the slice on the QUAD */
    glBegin(GL_QUADS);
        glVertex2f(0, 0);		glTexCoord2f(0, 0);
        glVertex2f(0, 1);		glTexCoord2f(1, 0);
        glVertex2f(1, 1);		glTexCoord2f(1, 1);
        glVertex2f(1, 0);		glTexCoord2f(0, 1);
    glEnd();
    glPopMatrix();

    /* @ Release texture reference & disable texturing */
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);

    /* @ Swapping buffer contents */
    glutSwapBuffers();
}

void OpenGL::ReshapeGL(int iWinWidth, int iWinHeight)
{
    // Adjusting viewPort
    glViewport(0, 0, iWinWidth, iWinHeight);

    // Adjusting window size
    glWinWidth = iWinWidth;
    glWinHeight = iWinHeight;

    // Load I to the projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}

void OpenGL::IdleGL()
{
    glutPostRedisplay();
}

void OpenGL::KeyboardGL(unsigned char fKey, int fX, int fY)
{

    // Dummy
    if (fX | fY | fKey) {}

    switch(fKey)
    {
        /* @ exit key */
        case 27:
            INFO("EXETING");
            EXIT(0);
            break;
        default:
            break;
    }


    glutPostRedisplay();
}

void OpenGL::MouseGL(int fButton, int fState, int fX, int fY)
{
    if(fState == GLUT_DOWN)
    {
        if(fButton == GLUT_LEFT_BUTTON)
        {
            INFO("1");
        }
        else if(fButton == GLUT_MIDDLE_BUTTON)
        {
             INFO("12");
        }
        else if(fButton == GLUT_RIGHT_BUTTON)
        {
             printf("13");
        }
    }
    else
    {
        //alternate code
    }

    /* @ Reslice & redisplay */
    glutPostRedisplay();
}

void OpenGL::MouseMotionGL(int iX, int iY)
{
    // Dummy
    if (iX | iY) {}

    glutPostRedisplay();
}

void OpenGL::RegisterGLCallBacks()
{
    /* Registering OpenGL context callbacks*/
    INFO("Registerng OpenGL context callbacks");

    glutDisplayFunc(DisplayGL);
    glutKeyboardFunc(KeyboardGL);
    glutReshapeFunc(ReshapeGL);
    glutIdleFunc(IdleGL);
    glutMouseFunc(MouseGL);
    glutMotionFunc(MouseMotionGL);

    INFO("Registerng OpenGL context callbacks DONE");
}
