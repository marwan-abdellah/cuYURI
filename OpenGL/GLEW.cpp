#include "GLEW.h"
#include "Utilities/MACROS.h"
#include <glew.h>


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
