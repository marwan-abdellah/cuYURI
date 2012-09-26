#include "OpenGL.h"
//#include "OpenGLCheck.h"
#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"

#include "OpenGL/Glut.h"

#include <glut.h>
#include "GL/gl.h"


/*************/
/* @ EXTERNS */
/*************/
float eXRot_Glob;
float eYRot_Glob;
float eZRot_Glob;
float eZoomLevel_Glob ;
float eSliceTrans_Glob ;
float eNormValue_Glob;

/************/
/* @ LOCALS */
/************/
GLuint* cGL_ImageTexture_ID;

int    eWinWidth;
int    eWinHeight;
float  eImageZoom       = 1;
float  eNormValue       = 1.0;
int    eGloWinWidth     = 512;
int    eGloWinHeight    = 512;

void OpenGL::updateSliceTexture(GLuint* iImageTexture_ID)
{
    cGL_ImageTexture_ID = iImageTexture_ID;
}

void OpenGL::prepareFBO(GLuint* iFBO_ID, GLuint* imageTex_ID)
{
    INFO("Preparing FBO");

   // glGenFramebuffersEXT(1, iFBO_ID);
   //// glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, *iFBO_ID);

    /* @ Attaching the FBO to the associated texture */
  //  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                           //   GL_TEXTURE_2D, *imageTex_ID, 0);

    /* @ Unbinding */
   /// glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
//
    INFO("Preparing FBO DONE");
}

void OpenGL::InitOpenGLContext(int argc, char** argv)
{
    INFO ("Initializing OpenGL Contex");

    /*
    // Checking the avialability of OpenGL context
    if (isOpenGLAvailable())
    {
        INFO("OpenGL device is available");
    }
    else
    {
        INFO("OpenGL device is NOT available");
        EXIT(0);
    }
    */

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

    INFO("Registering OpenGL callbacks");

    // Registering OpenGL CallBack Functions
    registerOpenGLCallBacks();

    INFO("Initializing OpenGL Contex DONE");
}




void OpenGL::InitOpenGL()
{
    /* @ Clearing color buffer */
    glClearColor (0.0, 0.0, 0.0, 0.0);

    /* Setting the pixel storage mode */
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
}

GLuint* OpenGL::AllocateTex(const int imageWidth,
                                  const int imageHeight,
                                  float* imagePtr)
{
    // Texture ID
    GLuint* imageTex_ID;

    INFO("1");
    //  Create 2D texture object as a render target
    glGenTextures(1, imageTex_ID);
    INFO("2");
    // Bind the texture target
    glBindTexture(GL_TEXTURE_2D, *imageTex_ID);
    INFO("3");
    // Set the 2D texture parameters
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    INFO("4");
    // @NOTE Automatic mipmap Generation included in OpenGL v1.4
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                 imageWidth, imageHeight, 0,
                 GL_LUMINANCE, GL_FLOAT, imagePtr);
    INFO("5");
    // Unbinding texture
    glBindTexture(GL_TEXTURE_2D, 0);
    INFO("6");
    return imageTex_ID;
}


GLuint* OpenGL::UploadImageToTexture(const int imageWidth,
                                  const int imageHeight,
                                  float* imagePtr, GLuint *imageTex_ID)
{
    // Texture ID
    GLuint* imageTex_ID_2;

    INFO("1");
    //  Create 2D texture object as a render target
    glGenTextures(1, imageTex_ID);
    INFO("2");
    // Bind the texture target
    glBindTexture(GL_TEXTURE_2D, *imageTex_ID);

    INFO("3");
    // Set the 2D texture parameters
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    INFO("4");
    // @NOTE Automatic mipmap Generation included in OpenGL v1.4
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                 imageWidth, imageHeight, 0,
                 GL_LUMINANCE, GL_FLOAT, imagePtr);
    INFO("5");
    // Unbinding texture
    glBindTexture(GL_TEXTURE_2D, 0);
    INFO("6");
    return imageTex_ID_2;
}

void OpenGL::displayGL()
{
    /* @ Clearing color buffer */
    glClear(GL_COLOR_BUFFER_BIT);

    /* @ Disabling depth test */
    glDisable(GL_DEPTH_TEST);
    /*
//    /* @ Binding slice texture to be displayed On OpenGL Quad */
//    glBindTexture(GL_TEXTURE_2D, *cGL_ImageTexture_ID);
//    glEnable(GL_TEXTURE_2D);

//    /* Adjusting slice texture parameters */
//    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
//    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

//    /* @ Adjusting viewport */
//    glViewport(-eWinWidth / 2, -eWinHeight / 2, eWinWidth * 2, eWinHeight * 2);

//    glMatrixMode(GL_MODELVIEW);
//    glPushMatrix();

//    /* @ Center slice texture at the orgin (0,0) */
//    glScalef(eImageZoom, eImageZoom, 1);
//    glTranslatef(-0.5, -0.5, 0.0);

//    /* @ Texture the slice on the QUAD */
//    glBegin(GL_QUADS);
//        glVertex2f(0, 0);		glTexCoord2f(0, 0);
//        glVertex2f(0, 1);		glTexCoord2f(1, 0);
//        glVertex2f(1, 1);		glTexCoord2f(1, 1);
//        glVertex2f(1, 0);		glTexCoord2f(0, 1);
//    glEnd();
//    glPopMatrix();

//    /* @ Release texture reference & disable texturing */
//    glBindTexture(GL_TEXTURE_2D, 0);
//    glDisable(GL_TEXTURE_2D);

//    /* @ Swapping buffer contents */
//    glutSwapBuffers();
}

void OpenGL::reshapeGL(int iWinWidth, int iWinHeight)
{
    /* @ Adjusting viewPort */
    glViewport(0, 0, iWinWidth, iWinHeight);

    /* @ For adjusting window size */
    eWinHeight = iWinHeight;
    eWinWidth = iWinWidth;

    // Projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
}

void OpenGL::idleGL()
{
    glutPostRedisplay();
}

void OpenGL::keyboardGL(unsigned char fKey, int fX, int fY)
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

void OpenGL::mouseGL(int fButton, int fState, int fX, int fY)
{
    if(fState == GLUT_DOWN)
    {
        if(fButton == GLUT_LEFT_BUTTON)
        {
            printf("1");
        }
        else if(fButton == GLUT_MIDDLE_BUTTON)
        {
             printf("12");
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

void OpenGL::mouseMotionGL(int iX, int iY)
{
    // Dummy
    if (iX | iY) {}

    glutPostRedisplay();
}

void OpenGL::registerOpenGLCallBacks()
{
    /* Registering OpenGL context callbacks*/
    INFO("Registerng OpenGL context callbacks");

    glutDisplayFunc(displayGL);
    glutKeyboardFunc(keyboardGL);
    glutReshapeFunc(reshapeGL);
    glutIdleFunc(idleGL);
    glutMouseFunc(mouseGL);
    glutMotionFunc(mouseMotionGL);

    INFO("Registerng OpenGL context callbacks DONE");
}
