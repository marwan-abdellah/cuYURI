#include "OpenGL.h"
#include "Cg.h"
#include "Utilities/MACROS.h"


// OpenGL internal parameters
int iRotX = 0;
int iRotY = 0;
int iRotZ = 0;

int iScaleX = 1;
int iScaleY = 1;
int iScaleZ = 1;

int iTransX = 0;
int iTransY = 0;
int iTransZ = 0;

int showColorCube = 0;




// Internal Cg context
CGcontext iCgContext;

// Internal shader profiles
CGprofile iVertProf;
CGprofile iFragProf;

// Internal shader programs
CGprogram iVertProg;
CGprogram iFragProg;

// internal Cg parameters
CGparameter iCgParam_1;
CGparameter iCgParam_2;


/*
void OpenGL::RayCaster::UpdateCgParamters(CGparameter CgParam1,
                                          CGparameter CgParam2)
{
    iCgParam_1 = CgParam1;
    iCgParam_2 = CgParam2;
}


void OpenGL::RayCaster::UpdateCgPrograms(CGprogram CgVertShader,
                                         CGprogram CgFragShader)
{
    iVertProg = CgVertShader;
    iFragProg = CgFragShader;
}

void OpenGL::RayCaster::UpdateCgProfiles(CGprofile CgVertProfile,
                                         CGprofile CgFragProfile)
{
    iVertProf = CgVertProfile;
    iFragProf = CgFragProfile;
}
*/

void OpenGL::RayCaster::EnableFrameBuffer(GLuint* frameBuffer)
{
    // Bind the frame buffer
   glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, *frameBuffer);
}

void OpenGL::RayCaster::EnableRenderBuffer(GLuint* renderBuffer)
{
    // Bind the specified render buffer
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, *renderBuffer);
}

void OpenGL::RayCaster::DisableFrameBuffer()
{
    // Unbind the frame buffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void OpenGL::RayCaster::CreateVertexWithColor(float x, float y, float z)
{
    // Generate vertex color RGB
    glColor3f(x, y, z);

    // Adjust texture corrdinate depnding on vertex color
    glMultiTexCoord3fARB(GL_TEXTURE1_ARB, x, y, z);

    // Draw vertex in position
    glVertex3f(x, y, z);
}

void OpenGL::RayCaster::DrawColoredCube(float x, float y, float z)
{
    glBegin(GL_QUADS);

    // Back side
    glNormal3f(0.0, 0.0, -1.0);
    OpenGL::RayCaster::CreateVertexWithColor(0.0, 0.0, 0.0);
    OpenGL::RayCaster::CreateVertexWithColor(0.0, y, 0.0);
    OpenGL::RayCaster::CreateVertexWithColor(x, y, 0.0);
    OpenGL::RayCaster::CreateVertexWithColor(x, 0.0, 0.0);

    // Front side
    glNormal3f(0.0, 0.0, 1.0);
    OpenGL::RayCaster::CreateVertexWithColor(0.0, 0.0, z);
    OpenGL::RayCaster::CreateVertexWithColor(x, 0.0, z);
    OpenGL::RayCaster::CreateVertexWithColor(x, y, z);
    OpenGL::RayCaster::CreateVertexWithColor(0.0, y, z);

    // Top side
    glNormal3f(0.0, 1.0, 0.0);
    OpenGL::RayCaster::CreateVertexWithColor(0.0, y, 0.0);
    OpenGL::RayCaster::CreateVertexWithColor(0.0, y, z);
    OpenGL::RayCaster::CreateVertexWithColor(x, y, z);
    OpenGL::RayCaster::CreateVertexWithColor(x, y, 0.0);

    // Bottom side
    glNormal3f(0.0, -1.0, 0.0);
    OpenGL::RayCaster::CreateVertexWithColor(0.0, 0.0, 0.0);
    OpenGL::RayCaster::CreateVertexWithColor(x, 0.0, 0.0);
    OpenGL::RayCaster::CreateVertexWithColor(x, 0.0, z);
    OpenGL::RayCaster::CreateVertexWithColor(0.0, 0.0, z);

    // Left side
    glNormal3f(-1.0, 0.0, 0.0);
    OpenGL::RayCaster::CreateVertexWithColor(0.0, 0.0, 0.0);
    OpenGL::RayCaster::CreateVertexWithColor(0.0, 0.0, z);
    OpenGL::RayCaster::CreateVertexWithColor(0.0, y, z);
    OpenGL::RayCaster::CreateVertexWithColor(0.0, y, 0.0);

    // Right side
    glNormal3f(1.0, 0.0, 0.0);
    OpenGL::RayCaster::CreateVertexWithColor(x, 0.0, 0.0);
    OpenGL::RayCaster::CreateVertexWithColor(x, y, 0.0);
    OpenGL::RayCaster::CreateVertexWithColor(x, y, z);
    OpenGL::RayCaster::CreateVertexWithColor(x, 0.0, z);

    glEnd();
}

void OpenGL::RayCaster::DrawFullScreenQuad()
{
    // Disabling depthe tests
    glDisable(GL_DEPTH_TEST);

    // Plot texture on the QUAD
    glBegin(GL_QUADS);
        glTexCoord2f(0, 0);     glVertex2f(0, 0);
        glTexCoord2f(1, 0);     glVertex2f(1, 0);
        glTexCoord2f(1, 1);     glVertex2f(1, 1);
        glTexCoord2f(0, 1);     glVertex2f(0, 1);
    glEnd();

    // Enable depth testing again
    glEnable(GL_DEPTH_TEST);
}


GLuint* OpenGL::RayCaster::UploadVolumeRGB(volumeImage* ptrVolImage)
{



    return NULL;

}

GLuint* OpenGL::RayCaster::UploadVolumeRGBA(volumeImage* ptrVolImage)
{
    INFO("Uploading the volume to the GPU");

    // Pixel data storage format
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);

    INFO("1");

    // Generate texture ID
    GLuint* volumeTex_ID = MEM_ALLOC_1D_GENERIC(GLuint, 1);

    INFO("2");
    // Generate texture ID
    glGenTextures(1, volumeTex_ID);

    INFO("3");
    // Bind texture
    glBindTexture(GL_TEXTURE_3D, *volumeTex_ID);
INFO("4");
    // Replacement mode
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    INFO("5");

    // Adjust 3D texture paramters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

    INFO("6");
    // Upload the texture to the GPU
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA,
                 ptrVolImage->volSize.NX,
                 ptrVolImage->volSize.NY,
                 ptrVolImage->volSize.NZ,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 ptrVolImage->volPtr);
    INFO("7");

    INFO("Uploading volume to 3D texture DONE");

    return volumeTex_ID;
}

GLuint* OpenGL::RayCaster::CreateFBO()
{
    INFO("Creating FBO for rendering the back face of the cube");

    GLuint* FBO_ID = MEM_ALLOC_1D_GENERIC(GLuint, 1);

    // Generate FBO ID
    glGenFramebuffersEXT(1, FBO_ID);

    // Bind the FBO
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, *FBO_ID);

    return FBO_ID;
}

GLuint* OpenGL::RayCaster::CreateBackFaceBuffer(const int WINDOW_SIZE_X,
                                                const int WINDOW_SIZE_Y)
{
    INFO("Creating buffer for rendering the back face of the cube");

    GLuint* backfaceTex_ID = MEM_ALLOC_1D_GENERIC(GLuint, 1);

    // Generate the texture ID
    glGenTextures(1, backfaceTex_ID);

    // Bind the backface texture
    glBindTexture(GL_TEXTURE_2D, *backfaceTex_ID);

    // Adjust texture parameters
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    // Allocate the texture required for the back face
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB,
                 WINDOW_SIZE_X, WINDOW_SIZE_Y,
                 0, GL_RGBA, GL_FLOAT, NULL);

    // Attachig the texture to the buffer for filling it with buffer data
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D, *backfaceTex_ID, 0);
    return backfaceTex_ID;
}


GLuint* OpenGL::RayCaster::CreateImageBuffer(const int WINDOW_SIZE_X,
                                             const int WINDOW_SIZE_Y)
{
    INFO("Creating buffer for rendering the final image");

    GLuint* imgTex_ID = MEM_ALLOC_1D_GENERIC(GLuint, 1);

    // Generate the texture ID
    glGenTextures(1, imgTex_ID);

    // Bind the texture
    glBindTexture(GL_TEXTURE_2D, *imgTex_ID);

    // Adjust texture parameters
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

     // Allocate the texture required for the rendering the final image
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA16F_ARB, WINDOW_SIZE_X, WINDOW_SIZE_Y, 0, GL_RGBA, GL_FLOAT, NULL);

    return imgTex_ID;
}

GLuint* OpenGL::RayCaster::CreateRenderBuffer(const int WINDOW_SIZE_X,
                                             const int WINDOW_SIZE_Y)
{

    INFO("Creating the rendering buffer");

    GLuint* renderBuffer_ID = MEM_ALLOC_1D_GENERIC(GLuint, 1);

    // Generate the render buffer ID
    glGenRenderbuffersEXT(1, renderBuffer_ID);

    // Bnd the render buffer
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, *renderBuffer_ID);

    // Setup the parameters of the render buffer
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT,
                             WINDOW_SIZE_X, WINDOW_SIZE_Y);

    // Attche the render buffer to the frame buffer
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                                 GL_RENDERBUFFER_EXT, *renderBuffer_ID);

    // Unbind the render buffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    INFO("1")
    return renderBuffer_ID;
}

// render the backface to the offscreen buffer backface_buffer
void OpenGL::RayCaster::RenderBackFace(GLuint* backfaceTex_ID)
{
    // Attach backface texture to the target
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D, *backfaceTex_ID, 0);

    // Clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // Enable culling
    glEnable(GL_CULL_FACE);

    // Cull the front face
    glCullFace(GL_FRONT);

    // Do the color mapping in the cube
    OpenGL::RayCaster::DrawColoredCube(1.0, 1.0, 1.0);

    // Disable culling
    glDisable(GL_CULL_FACE);
}


// display the final image on the screen
void OpenGL::RayCaster::RenderBufferToScreen(GLuint* imageTex_ID, GLuint* backfaceTex_ID,
                                             const int WINDOW_SIZE_X,
                                             const int WINDOW_SIZE_Y)
{
    // Clearing bufers
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // Loading I to the MV matrix
    glLoadIdentity();

    // Enable 2D texturing
    glEnable(GL_TEXTURE_2D);

    if(!showColorCube)
        // Show the volume dataset
        glBindTexture(GL_TEXTURE_2D, *imageTex_ID);
    else
        // Show the cube
        glBindTexture(GL_TEXTURE_2D, *backfaceTex_ID);

    // Reshaping the projection according to the window size
    OpenGL::RayCaster::ReshapeProjection(WINDOW_SIZE_X, WINDOW_SIZE_Y);

    // Map the texture to the proxy geometry
    OpenGL::RayCaster::DrawFullScreenQuad();

    // Disable texturing
    glDisable(GL_TEXTURE_2D);
}


void OpenGL::RayCaster::ReshapeProjection(int projectionWidth, int projectionHeight)
{
    if (projectionHeight == 0)
        projectionHeight = 1;

    // Set the new viewport
    glViewport(0, 0, projectionWidth, projectionHeight);

    // Set the projection matrix in the matrix stack
    glMatrixMode(GL_PROJECTION);

    // Load I to the projection matrix
    glLoadIdentity();

    // Orithograohic projectio
    gluOrtho2D(0, 1, 0, 1);

    // Load again the model view matrix in the stack
    glMatrixMode(GL_MODELVIEW);
}




/*
void RayCasting_SinglePass(GLuint* imageTex_ID,)
{
    // Attach final image texture to frame buffer
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D, *final_image, 0);

    // Clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Enable shading profiles
    cgGLEnableProfile(vertexProfile);
    cgGLEnableProfile(fragmentProfile);

    // Bind the shader programs
    cgGLBindProgram(vertex_main);
    cgGLBindProgram(fragment_main);

    // Bind the parameters to the shader
    cgGLSetParameter1f(cgGetNamedParameter(fragment_main, "stepsize") , stepsize);

    // Set shader uniform parameters
    OpenGL::Cg::SetTextureUniformParams("tex",*backface_buffer,fragment_main,param1);
    OpenGL::Cg::SetTextureUniformParams("volume_tex",*volume_texture,fragment_main,param2);

    // Enable culling
    glEnable(GL_CULL_FACE);

    // Cull the back face of the cube
    glCullFace(GL_BACK);

    // Draw the color coded cube
    OpenGL::RayCaster::DrawColoredCube(1.0, 1.0, 1.0);

    // Disabling culling
    glDisable(GL_CULL_FACE);

    // Disable shading profiles
    cgGLDisableProfile(vertexProfile);
    cgGLDisableProfile(fragmentProfile);
}
*/
