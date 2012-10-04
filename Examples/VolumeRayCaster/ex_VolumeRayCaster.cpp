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

#include "ex_VolumeRayCaster.h"
#include "CUDA/cuGlobals.h"
#include "CUDA/cuUtilities.h"
#include "CUDA/cuYURI_Interfaces.h"
#include "Globals.h"
#include "Utilities/Utils.h"
#include "Utilities/MACROS.h"
#include "Utilities/MemoryMACROS.h"

#include "OpenGL/GLEW.h"

#include <glew.h>
#include <Cg/cg.h>
#include <fstream>
#include <Cg/cgGL.h>
#include <GL/glut.h>
#include <iostream>

#include <sstream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cassert>
#include "Vector3.h"

#include "OpenGL/Cg.h"
#include "OpenGL/OpenGL.h"
#include "Volume/Volume.h"

#define MAX_KEYS 256
#define WINDOW_SIZE 800
#define VOLUME_TEX_SIZE 128

using namespace std;

// Globals ------------------------------------------------------------------

bool gKeys[MAX_KEYS];
bool toggle_visuals = true;
CGcontext context;
CGprofile vertexProfile, fragmentProfile;
CGparameter cgParam_1,cgParam_2;

CGprogram vertex_main,fragment_main; // the raycasting shader programs

GLuint* renderBuffer;
GLuint* frameBuffer;
GLuint* volTex_ID;
GLuint *backfacrBuffTex_ID;
GLuint *imageTex_ID;

float stepsize = 1.0/50.0;


void init()
{
    // Chekc GLEW extensions
    OpenGL::CheckGLExtensionsRequiredForRayCaster();

    // Enable culling
    glEnable(GL_CULL_FACE);

    // Clear color buffer
    glClearColor(0.0, 0.0, 0.0, 0);

    // Load volume (Create sample volume)
    volumeImage* volImagePtr = MEM_ALLOC_1D_GENERIC(volumeImage, 1);
    volImagePtr = Volume::CreateTestVolume(VOLUME_TEX_SIZE);

    // 3D texture ID
    volTex_ID= OpenGL::RayCaster::UploadVolumeRGBA(volImagePtr);

    // Freeing the volume data
    free(volImagePtr->volPtr);

    // CG init
    cgSetErrorCallback(OpenGL::Cg::cgErrorCallback);

    // Create Cg context
    context = cgCreateContext();

    // Get vertex and fragment profiles
    vertexProfile = OpenGL::Cg::GetVertexProfile(context);
    fragmentProfile = OpenGL::Cg::GetFragmentProfile(context);

    // Load vertex program and check for erros
    OpenGL::Cg::LoadVertexProgram(context, vertexProfile, vertex_main,"raycasting_shader.cg","vertex_main");
    OpenGL::Cg::cgErrorCallback(context);

    // Load fragment program and check for errors
    OpenGL::Cg::LoadFragmentProgram(context, fragmentProfile, fragment_main,"raycasting_shader.cg","fragment_main");
    OpenGL::Cg::cgErrorCallback(context);

    // Create FBO
    frameBuffer = OpenGL::RayCaster::CreateFBO();

    // Create back buffer
    backfacrBuffTex_ID = OpenGL::RayCaster::CreateBackFaceBuffer(WINDOW_SIZE, WINDOW_SIZE);

    // Create image buffer
    imageTex_ID = OpenGL::RayCaster::CreateImageBuffer(WINDOW_SIZE, WINDOW_SIZE);

    // Create render buffer
    renderBuffer = OpenGL::RayCaster::CreateRenderBuffer(WINDOW_SIZE,WINDOW_SIZE);
}


// for contiunes keypresses
void ProcessKeys()
{
    // Process keys
    for (int i = 0; i < 256; i++)
    {
        if (!gKeys[i])  { continue; }
        switch (i)
        {
        case ' ':
            break;
        case 'w':
            stepsize += 1.0/2048.0;
            if(stepsize > 0.25) stepsize = 0.25;
            break;
        case 'e':
            stepsize -= 1.0/2048.0;
            if(stepsize <= 1.0/200.0) stepsize = 1.0/200.0;
            break;
        }
    }

}

void key(unsigned char k, int x, int y)
{
    gKeys[k] = true;
}

void KeyboardUpCallback(unsigned char key, int x, int y)
{
    gKeys[key] = false;

    switch (key)
    {
    case 27 :
        {
            exit(0); break;
        }
    case ' ':
        toggle_visuals = !toggle_visuals;
        break;
    }
}

// glut idle function
void idle_func()
{
    ProcessKeys();
    glutPostRedisplay();
}

void reshape_ortho(int w, int h)
{
    if (h == 0) h = 1;
    glViewport(0, 0,w,h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
}


void resize(int w, int h)
{
    if (h == 0) h = 1;
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)w/(GLfloat)h, 0.01, 400.0);
    glMatrixMode(GL_MODELVIEW);
}


// display the final image on the screen
void render_buffer_to_screen()
{
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    if(toggle_visuals)
        glBindTexture(GL_TEXTURE_2D,*imageTex_ID);
    else
        glBindTexture(GL_TEXTURE_2D,*backfacrBuffTex_ID);
    reshape_ortho(WINDOW_SIZE,WINDOW_SIZE);
     OpenGL::RayCaster::DrawFullScreenQuad();
    glDisable(GL_TEXTURE_2D);
}



void raycasting_pass()
{
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, *imageTex_ID, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    cgGLEnableProfile(vertexProfile);
    cgGLEnableProfile(fragmentProfile);
    cgGLBindProgram(vertex_main);
    cgGLBindProgram(fragment_main);
    cgGLSetParameter1f( cgGetNamedParameter( fragment_main, "stepsize") , stepsize);
    OpenGL::Cg::SetTextureUniformParams("tex",*backfacrBuffTex_ID,fragment_main,cgParam_1);
    OpenGL::Cg::SetTextureUniformParams("volume_tex",*volTex_ID,fragment_main,cgParam_2);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    // drawQuads(1.0,1.0, 1.0);

    OpenGL::RayCaster::DrawColoredCube(1.0, 1.0, 1.0);
    glDisable(GL_CULL_FACE);
    cgGLDisableProfile(vertexProfile);
    cgGLDisableProfile(fragmentProfile);
}

// This display function is called once pr frame
void display()
{
    static float rotate = 0;
    rotate += 0.25;

    resize(WINDOW_SIZE,WINDOW_SIZE);

    // Enable rendering buffers
    OpenGL::RayCaster::EnableRenderBuffer(renderBuffer);
    OpenGL::RayCaster::EnableFrameBuffer(frameBuffer);


    // Load I to the model view matrix
    glLoadIdentity();

    // Translate the cube along the Z direction
    glTranslatef(0,0,-2.0);

    // Rotation of the volume around the YZ
    glRotatef(rotate, 0, 1, 1);

    // Centering the texture cube
    glTranslatef(-0.5, -0.5, -0.5);

    OpenGL::RayCaster::RenderBackFace(backfacrBuffTex_ID);


    ///////
    //update the parametewrs

    raycasting_pass();

    // Disabling frame buffer
    OpenGL::RayCaster::DisableFrameBuffer();

    render_buffer_to_screen();
    glutSwapBuffers();
}

void ex::VolumeRayCaster::run(int argc, char** argv)
{
    INFO("Displaying a sample float array on OpenGL Widget");
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("GPU raycasting tutorial");

    glutReshapeWindow(WINDOW_SIZE,WINDOW_SIZE);

    glutKeyboardFunc(key);
    glutKeyboardUpFunc(KeyboardUpCallback);

    glutDisplayFunc(display);
    glutIdleFunc(idle_func);
    glutReshapeFunc(resize);
    resize(WINDOW_SIZE,WINDOW_SIZE);
    init();
    glutMainLoop();
    return 0;

    INFO("DONE");
}
