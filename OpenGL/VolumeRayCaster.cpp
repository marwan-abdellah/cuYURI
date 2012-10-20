#include "VolumeRayCaster.h"
#include "Volume/Volume.h"
#include "Volume/Loader.h"

#include "Utilities/MACROS.h"

#define MAX_KEYS 256
#define WINDOW_SIZE 800

using namespace std;

bool gKeys[MAX_KEYS];

bool ToggleColorCube = true;

// Cg context
CGcontext cgContext;

// Vertex & Fragment profiles
CGprofile vertexProfile;
CGprofile fragmentProfile;

// Shading parameters
CGparameter cgParam_1;
CGparameter cgParam_2;

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


namespace VolRayCaster
{

void cgErrorCallback()
{
    INFO("Checking CG callbacks errors");

    CGerror cgError = cgGetError();

    if(cgError)
    {
        INFO (cgGetErrorString(cgError));

        if(cgContext != NULL)
            INFO(CATS("Last Cg cgContext : ") + cgGetLastListing(cgContext));

        // Exiting application
        EXIT(0);
    }
}

// Sets a uniform texture parameter
void SetUnformTexParameter(char* par, GLuint tex,const CGprogram &program,CGparameter param)
{
    param = cgGetNamedParameter(program, par);
    cgGLSetTextureParameter(param, tex);
    cgGLEnableTextureParameter(param);
}


// LoadVertexProgram: loading a vertex program
void LoadVertexProgram(CGprogram &v_program,char *shader_path, char *program_name)
{
    assert(cgIsContext(cgContext));
    v_program = cgCreateProgramFromFile(cgContext, CG_SOURCE,shader_path,
        vertexProfile,program_name, NULL);
    if (!cgIsProgramCompiled(v_program))
        cgCompileProgram(v_program);

    cgGLEnableProfile(vertexProfile);
    cgGLLoadProgram(v_program);
    cgGLDisableProfile(vertexProfile);
}

// LoadFragmentProgram: loading a fragment program
void LoadFragmentProgram(CGprogram &f_program,char *shader_path, char *program_name)
{

    assert(cgIsContext(cgContext));
    f_program = cgCreateProgramFromFile(cgContext, CG_SOURCE, shader_path,
        fragmentProfile,program_name, NULL);
    if (!cgIsProgramCompiled(f_program))
        cgCompileProgram(f_program);

    cgGLEnableProfile(fragmentProfile);
    cgGLLoadProgram(f_program);
    cgGLDisableProfile(fragmentProfile);
}

void EnableRenderBuffers()
{
    glBindFramebufferEXT (GL_FRAMEBUFFER_EXT, frameBuffer);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderBuffer);
}

void DisableRenderBuffers()
{
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void CreateVertexWithColor(float x, float y, float z)
{
    // Generate vertex color RGB
    glColor3f(x, y, z);

    // Adjust texture corrdinate depnding on vertex color
    glMultiTexCoord3fARB(GL_TEXTURE1_ARB, x, y, z);

    // Draw vertex in position
    glVertex3f(x, y, z);
}

// this method is used to draw the front and backside of the volume
void DrawColoredCube(float x, float y, float z)
{
    glBegin(GL_QUADS);

        // Back side
        glNormal3f(0.0, 0.0, -1.0);
        CreateVertexWithColor(0.0, 0.0, 0.0);
        CreateVertexWithColor(0.0, y, 0.0);
        CreateVertexWithColor(x, y, 0.0);
        CreateVertexWithColor(x, 0.0, 0.0);

        // Front side
        glNormal3f(0.0, 0.0, 1.0);
        CreateVertexWithColor(0.0, 0.0, z);
        CreateVertexWithColor(x, 0.0, z);
        CreateVertexWithColor(x, y, z);
        CreateVertexWithColor(0.0, y, z);

        // Top side
        glNormal3f(0.0, 1.0, 0.0);
        CreateVertexWithColor(0.0, y, 0.0);
        CreateVertexWithColor(0.0, y, z);
        CreateVertexWithColor(x, y, z);
        CreateVertexWithColor(x, y, 0.0);

        // Bottom side
        glNormal3f(0.0, -1.0, 0.0);
        CreateVertexWithColor(0.0, 0.0, 0.0);
        CreateVertexWithColor(x, 0.0, 0.0);
        CreateVertexWithColor(x, 0.0, z);
        CreateVertexWithColor(0.0, 0.0, z);

        // Left side
        glNormal3f(-1.0, 0.0, 0.0);
        CreateVertexWithColor(0.0, 0.0, 0.0);
        CreateVertexWithColor(0.0, 0.0, z);
        CreateVertexWithColor(0.0, y, z);
        CreateVertexWithColor(0.0, y, 0.0);

        // Right side
        glNormal3f(1.0, 0.0, 0.0);
        CreateVertexWithColor(x, 0.0, 0.0);
        CreateVertexWithColor(x, y, 0.0);
        CreateVertexWithColor(x, y, z);
        CreateVertexWithColor(x, 0.0, z);

    glEnd();

}

// create a test volume texture, here you could load your own volume
void CreateVolumeTexture()
{
    INFO("Creating a sample volume");

    // Load volume (create sample volume)
    volumeImage* volImagePtr = MEM_ALLOC_1D_GENERIC(volumeImage, 1);
     volImagePtr = Volume::CreateTestVolume(256);
    //volImagePtr = Volume::LoadVolume("/home/abdellah/Software/DataSets/CTData/CTData");

    INFO("Uploading volume texture to the GPU");

    // Upload volume to a GPU texture
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glGenTextures(1, &volumeTex_ID);
    glBindTexture(GL_TEXTURE_3D, volumeTex_ID);
    // glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

    glTexImage3D(GL_TEXTURE_3D, 0,GL_RGBA,
                 volImagePtr->volSize.NX,
                 volImagePtr->volSize.NY,
                 volImagePtr->volSize.NZ,
                 0, GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 volImagePtr->volPtr);

/*    glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE,
                 volImagePtr->volSize.NX,
                 volImagePtr->volSize.NY,
                 volImagePtr->volSize.NZ,
                 0, GL_LUMINANCE,
                 GL_UNSIGNED_BYTE,
                 volImagePtr->volPtr)*/;

    INFO("Volume texture created");
}


void Init()
{
    INFO("Initializing GLEW");
    GLenum glewError = glewInit();

    // Initialize all the OpenGL extensions
    glewGetExtension("glMultiTexCoord2fvARB");

    if(glewGetExtension("GL_EXT_frameBuffer_object"))
        INFO("GL_EXT_frameBuffer_object support");

    if(glewGetExtension("GL_EXT_renderBuffer_object"))
        INFO("GL_EXT_renderBuffer_object support");

    if(glewGetExtension("GL_ARB_vertex_buffer_object"))
        INFO("GL_ARB_vertex_buffer_object support");

    if(GL_ARB_multitexture)
        INFO("GL_ARB_multitexture support");

    if (glewGetExtension("GL_ARB_fragment_shader")      != GL_TRUE ||
        glewGetExtension("GL_ARB_vertex_shader")        != GL_TRUE ||
        glewGetExtension("GL_ARB_shader_objects")       != GL_TRUE ||
        glewGetExtension("GL_ARB_shading_language_100") != GL_TRUE)
    {
        INFO("Driver does not support OpenGL Shading Language");
        INFO("EXITING ... ");
        EXIT(0);
    }

    glEnable(GL_CULL_FACE);

    // Clearing color buffer
    glClearColor(0.0, 0.0, 0.0, 0);

    // Creating volume texture and uploading it to the GPU
    CreateVolumeTexture();

    // Initialization Cg cgContext
    cgSetErrorCallback(cgErrorCallback);
    cgContext = cgCreateContext();

    // Checking Cg vertex profiles
    if (cgGLIsProfileSupported(CG_PROFILE_VP40))
    {
        vertexProfile = CG_PROFILE_VP40;
        INFO("CG_PROFILE_VP40 supported");
    }
    else
    {
        if (cgGLIsProfileSupported(CG_PROFILE_ARBVP1))
            vertexProfile = CG_PROFILE_ARBVP1;
        else
        {
            INFO("Neither arbvp1 or vp40 vertex profiles supported on this system");
            INFO("EXITING ... ");
            EXIT(0);
        }
    }

    // Checking Cg fragment profiles
    if (cgGLIsProfileSupported(CG_PROFILE_FP40))
    {
        fragmentProfile = CG_PROFILE_FP40;
        INFO("CG_PROFILE_FP40 supported");
    }
    else
    {
        if (cgGLIsProfileSupported(CG_PROFILE_ARBFP1))
            fragmentProfile = CG_PROFILE_ARBFP1;
        else
        {
            INFO("Neither arbfp1 or fp40 fragment profiles supported on this system");
            INFO("EXITING ... ");
            EXIT(0);
        }
    }

    // Load shaders
    LoadVertexProgram(vertexProgram,"raycasting_shader.cg","vertex_main");
    //cgErrorCallback();
    LoadFragmentProgram(fragmentProgram,"raycasting_shader.cg","fragment_main");
    //cgErrorCallback();


    INFO("PASS");
    // Create buffers
    glGenFramebuffersEXT(1, &frameBuffer);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,frameBuffer);

    glGenTextures(1, &backfaceBuffer);
    glBindTexture(GL_TEXTURE_2D, backfaceBuffer);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA16F_ARB, WINDOW_SIZE, WINDOW_SIZE,
                 0, GL_RGBA, GL_FLOAT, NULL);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D, backfaceBuffer, 0);

    glGenTextures(1, &imageTex_ID);
    glBindTexture(GL_TEXTURE_2D, imageTex_ID);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA16F_ARB, WINDOW_SIZE, WINDOW_SIZE,
                 0, GL_RGBA, GL_FLOAT, NULL);

    glGenRenderbuffersEXT(1, &renderBuffer);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderBuffer);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT,
                             WINDOW_SIZE, WINDOW_SIZE);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                                 GL_RENDERBUFFER_EXT, renderBuffer);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

}

void Keyboard_GL(unsigned char fKey, int fX, int fY)
{
    if (fX | fY | fKey) {}

    switch(fKey)
    {
    // Exit key
    case 27:
        INFO("EXETING");
        EXIT(0);
        break;

    // Toggling the color cube and the volume
    case ' ':
        ToggleColorCube = !ToggleColorCube;
        break;

    // Increase the sampling step
    case 'w':
        samplingStep += 1.0 / 2048.0;
        if(samplingStep > 0.25)
            samplingStep = 0.25;
        break;

    // Decrease the sampling step
    case 'e':
        samplingStep -= 1.0 / 2048.0;
        if(samplingStep <= 1.0 / 200.0)
            samplingStep= 1.0 / 200.0;
        break;

    // Increase the scaling value
    case 'r':
        zoomValue *= 1.1;
        break;

    // Decrease the scaling value
    case 't':
        zoomValue /= 1.1;
        break;

    // X-axis rotation
    case 'a': rotateX += 5;
        break;
    case 'A': rotateX -= 5;
        break;

    // Y-axis rotation
    case 's': rotateY += 5;
        break;
    case 'S': rotateY -= 5;
        break;

    // Z-axis rotation
    case 'd': rotateZ += 5;
        break;
    case 'D': rotateZ -= 5;
        break;

    // Default case
    default:
        break;
    }
    glutPostRedisplay();
}

void Idel_GL()
{
    glutPostRedisplay();
}

void ReshapeOrtho_GL(int windowWidth, int windowHeight)
{
    if (windowHeight == 0)
        windowHeight = 1;

    // Adjust viewport
    glViewport(0, 0, windowWidth, windowHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Orthograhic projection
    gluOrtho2D(0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
}

void Resize_GL(int windowWidth, int windowHeight)
{
    if (windowHeight == 0)
        windowHeight = 1;

    // Adjust viewport
    glViewport(0, 0, windowWidth, windowHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    // Perspective projection
    gluPerspective(60.0, (GLfloat)windowWidth / (GLfloat)windowHeight, 0.01, 400.0);
    glMatrixMode(GL_MODELVIEW);
}

void RenderFinalImage()
{
    // Disable depth testing
    glDisable(GL_DEPTH_TEST);

    // Texture the quad with the final reconstructed image
    glBegin(GL_QUADS);
        glTexCoord2f(0,0);      glVertex2f(0,0);
        glTexCoord2f(1,0);      glVertex2f(1,0);
        glTexCoord2f(1, 1);     glVertex2f(1, 1);
        glTexCoord2f(0, 1);     glVertex2f(0, 1);
    glEnd();

    // Get back the depth testing
    glEnable(GL_DEPTH_TEST);
}

// display the final image on the screen
void RenderBufferToScreen()
{
    // Clearing buffers
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // Loading I
    glLoadIdentity();

    // Enable texturing
    glEnable(GL_TEXTURE_2D);

    // Render the loaded volume or the color cube
    if(ToggleColorCube)
        glBindTexture(GL_TEXTURE_2D, imageTex_ID);
    else
        glBindTexture(GL_TEXTURE_2D, backfaceBuffer);

    // Reshaping the window
    ReshapeOrtho_GL(WINDOW_SIZE, WINDOW_SIZE);

    // Render the final image
    RenderFinalImage();

    // Disable texturing
    glDisable(GL_TEXTURE_2D);
}

// render the backface to the offscreen buffer backfaceBuffer
void RenderBackface()
{
    // Attach texture to buffer
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D, backfaceBuffer, 0);

    // Clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // Enable culling
    glEnable(GL_CULL_FACE);

    // Cukking the front face
    glCullFace(GL_FRONT);

    // Draw the color bounding cube
    DrawColoredCube(1.0,1.0, 1.0);

    // Disable culling
    glDisable(GL_CULL_FACE);
}

void RayCastingPass()
{
    // Attach texture to buffer
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT,
                              GL_TEXTURE_2D,
                              imageTex_ID, 0);

    // Clearing buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Enable shading progfiles
    cgGLEnableProfile(vertexProfile);
    cgGLEnableProfile(fragmentProfile);

    // Bind the shaders to the GPU for execution
    cgGLBindProgram(vertexProgram);
    cgGLBindProgram(fragmentProgram);

    // Update the shader parameters
    cgGLSetParameter1f(cgGetNamedParameter(fragmentProgram, "samplingStep"), samplingStep);
    SetUnformTexParameter("tex", backfaceBuffer, fragmentProgram, cgParam_1);
    SetUnformTexParameter("volume_tex", volumeTex_ID, fragmentProgram, cgParam_2);

    // Enable culling
    glEnable(GL_CULL_FACE);

    // Cull the backface
    glCullFace(GL_BACK);

    // Draw the bounding colored cube
    DrawColoredCube(1.0,1.0, 1.0);

    // Disable the culling
    glDisable(GL_CULL_FACE);

    // Disable shader profiles
    cgGLDisableProfile(vertexProfile);
    cgGLDisableProfile(fragmentProfile);
}

// This display function is called once pr frame
void Display_GL()
{
    // Increment rotation
     rotateX += 0.25;
     rotateY += 0.25;
     rotateZ += 0.25;

    // Resize window
    Resize_GL(WINDOW_SIZE,WINDOW_SIZE);

    // Enable render buffer
    EnableRenderBuffers();

    glLoadIdentity();

    // Z-axix translation
    glTranslatef(0, 0, -2.25);

    glScalef(zoomValue, zoomValue, zoomValue);

    // X-axis rotation
    glRotatef(rotateX, 1, 0, 0);

    // Y-axis rotation
    glRotatef(rotateY, 0, 1, 0);

    // Z-axis rotation
    glRotatef(rotateZ, 0, 0, 1);

    // Centring the cube
    glTranslatef(-0.5, -0.5, -0.5);

    // Render the backface of the cube
    RenderBackface();

    // Raycast the volume
    RayCastingPass();

    // Disable the render buffers
    DisableRenderBuffers();

    // Render the final image to the texture
    RenderBufferToScreen();

    // Swap buffers
    glutSwapBuffers();
}

int VolRayCaster(int argc, char* argv[])
{
    INFO("Initializing Volume Raycasting");

    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("GPU Raycasting - cuYURI integration");
    glutReshapeWindow(WINDOW_SIZE,WINDOW_SIZE);

    glutKeyboardFunc(Keyboard_GL);
    glutDisplayFunc(Display_GL);
    glutIdleFunc(Idel_GL);
    glutReshapeFunc(Resize_GL);

    Resize_GL(WINDOW_SIZE,WINDOW_SIZE);

    Init();
    glutMainLoop();
    return 0;
}
}
