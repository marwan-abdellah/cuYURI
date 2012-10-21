#include "GLEW.h"
#include "MACROS/MACROS.h"
#include <GL/glew.h>


bool OpenGL::CheckGLExtensions()
{
    INFO("Checking OpenGL Extensions - GLEW");

    // Initializing GLEW
    glewInit();

    // Check OpenGL 2.0
    if (! glewIsSupported("GL_VERSION_2_0"))
    {
        INFO("ERROR: Support for necessary OpenGL extensions missing ");
        return 0;
    }
    else
        return 1;
}


void OpenGL::CheckGLExtensionsRequiredForRayCaster()
{
    cout << "glew init " << endl;
    GLenum err = glewInit();

    // initialize all the OpenGL extensions
    glewGetExtension("glMultiTexCoord2fvARB");
    if(glewGetExtension("GL_EXT_framebuffer_object") )cout << "GL_EXT_framebuffer_object support " << endl;
    if(glewGetExtension("GL_EXT_renderbuffer_object"))cout << "GL_EXT_renderbuffer_object support " << endl;
    if(glewGetExtension("GL_ARB_vertex_buffer_object")) cout << "GL_ARB_vertex_buffer_object support" << endl;
    if(GL_ARB_multitexture)cout << "GL_ARB_multitexture support " << endl;

    if (glewGetExtension("GL_ARB_fragment_shader")      != GL_TRUE ||
        glewGetExtension("GL_ARB_vertex_shader")        != GL_TRUE ||
        glewGetExtension("GL_ARB_shader_objects")       != GL_TRUE ||
        glewGetExtension("GL_ARB_shading_language_100") != GL_TRUE)
    {
        cout << "Driver does not support OpenGL Shading Language" << endl;
        exit(1);
    }
}

