#include "Cg.h"
#include <cassert>

void OpenGL::Cg::cgErrorCallback(CGcontext currentContext)
{
    // Get last error
    CGerror lastError = cgGetError();

    // Check for errors
    if(lastError)
    {
        INFO(cgGetErrorString(lastError));

        // Get currentContext error
        if(currentContext != NULL)
            INFO(cgGetLastListing(currentContext));

        // Exitting
        EXIT(0);
    }
}

// Sets a uniform texture paramName
void OpenGL::Cg::SetTextureUniformParams(char* paramName,
                                         GLuint iTex_ID,
                                         const CGprogram &shaderProg,
                                         CGparameter paramVal)
{
    // Get the shader uniform parameter
    paramVal = cgGetNamedParameter(shaderProg, paramName);

    // Set the parameter for a texture specified by the texure ID
    cgGLSetTextureParameter(paramVal, iTex_ID);

    // Enable the parameter
    cgGLEnableTextureParameter(paramVal);
}

// Loads a vertex program
void OpenGL::Cg::LoadVertexProgram(CGcontext currentContext,
                                   CGprofile vertexProfile,
                                   CGprogram &vertexProg,
                                   char *shaderFilePath, char *progName)
{
    // Make sure that the current Cg context is running
    assert(cgIsContext(currentContext));

    // Get the vertex program
    vertexProg = cgCreateProgramFromFile
            (currentContext, CG_SOURCE, shaderFilePath,
             vertexProfile, progName, NULL);


    // Compile the vertex program on the fly
    if (!cgIsProgramCompiled(vertexProg))
        cgCompileProgram(vertexProg);

    // Enable the shader profile
    cgGLEnableProfile(vertexProfile);

    // Load the shader program and apply it
    cgGLLoadProgram(vertexProg);

    // Disabling the shader profile
    cgGLDisableProfile(vertexProfile);
}

// Loading a fragment program
void OpenGL::Cg::LoadFragmentProgram(CGcontext currentContext,
                                     CGprofile fragmentProfile,
                                     CGprogram &fragmentProg,
                                     char *shaderFilePath, char *progName)
{
    // Make sure that the current Cg context is running
    assert(cgIsContext(currentContext));

    // Get the fragment program
    fragmentProg = cgCreateProgramFromFile
            (currentContext, CG_SOURCE, shaderFilePath,
             fragmentProfile, progName, NULL);

    // Compile the fragment program on the fly
    if (!cgIsProgramCompiled(fragmentProg))
        cgCompileProgram(fragmentProg);

    // Enable the shader profile
    cgGLEnableProfile(fragmentProfile);

    // Load the shader program and apply it
    cgGLLoadProgram(fragmentProg);

    // Disabling the shader profile
    cgGLDisableProfile(fragmentProfile);
}

CGprofile OpenGL::Cg::GetVertexProfile(CGcontext currentContext)
{
    CGprofile vertexProfile;

    // Cg context should be running
    if(currentContext != NULL)
    {
        // Check the existance of the CG_PROFILE_VP40 profile
        if (cgGLIsProfileSupported(CG_PROFILE_VP40))
        {
            vertexProfile = CG_PROFILE_VP40;

            INFO("CG_PROFILE_VP40 is supported");
        }
        else
        {
            // Check the existance of the CG_PROFILE_ARBVP1 profile
            if (cgGLIsProfileSupported(CG_PROFILE_ARBVP1))
                vertexProfile = CG_PROFILE_ARBVP1;
            else
            {
                INFO("Neither arbvp1 or vp40 vertex profiles supported on this system... EXITING");
                EXIT(0);
            }
        }
    }
    else
    {
        INFO("Cg context is not valid. Exiting");
        EXIT(0);
    }

    return vertexProfile;
}

CGprofile OpenGL::Cg::GetFragmentProfile(CGcontext currentContext)
{
    CGprofile fragmentProfile;

    // Cg context should be running
    if(currentContext != NULL)
    {
        // Check the existance of the CG_PROFILE_FP40 profile
        if (cgGLIsProfileSupported(CG_PROFILE_FP40))
        {
            fragmentProfile = CG_PROFILE_FP40;

            INFO("CG_PROFILE_FP40 is supported");
        }
        else
        {
            // Check the existance of the CG_PROFILE_ARBFP1 profile
            if (cgGLIsProfileSupported(CG_PROFILE_ARBFP1))
                fragmentProfile = CG_PROFILE_ARBFP1;
            else
            {
                INFO("Neither arbfp1 or fp40 fragment profiles supported on this system");
                EXIT(0);
            }
        }
    }
    else
    {
        INFO("Cg context is not valid. Exiting");
        EXIT(0);
    }

    return fragmentProfile;
}

