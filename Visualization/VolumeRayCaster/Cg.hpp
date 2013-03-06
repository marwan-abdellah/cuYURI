/*********************************************************************
 * Copyright © 2007-2012,
 * Marwan Abdellah: <abdellah.marwan@gmail.com>
 *
 * This code is part of the Ray Casting Tutorial provided by
 * Peter Trier <trier@daimi.au.dk>

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

#ifndef _CG_H_
#define _CG_H_

/*!
 * @ Interfaces
 */
#include "VolumeRayCaster.h"
#include "Parameters.h"

/*!
 * @ Implementations
 */
#include "ColorCube.hpp"
#include "GL_CallBacks.hpp"
#include "GL_Buffers.hpp"
#include "GLEW.hpp"
#include "GLUT.hpp"
#include "LoadingVolume.hpp"
#include "Rendering.hpp"
#include "VolumeData.hpp"

namespace RayCaster
{
void cgErrorCallback()
{
    INFO("Checking CG callbacks errors");

    CGerror cgError = cgGetError();

    if(cgError)
    {
        // Reporting the error string
        INFO (cgGetErrorString(cgError));

        // Reportin last Cg operation
        if(cgContext != NULL)
            INFO(CATS("Last Cg cgContext : ") + cgGetLastListing(cgContext));

        // Exiting application
        EXIT(0);
    }
}

void SetUnformTexParameter(char* shaderParam, GLuint iTex_ID,
                           const CGprogram &shaderProg,
                           CGparameter param)
{
    // Getting shader parameter w/ name
    param = cgGetNamedParameter(shaderProg, shaderParam);

    // Updating parameter
    cgGLSetTextureParameter(param, iTex_ID);

    // Enable the update
    cgGLEnableTextureParameter(param);
}

void LoadVertexProgram(CGprogram &vertexProg,
                       char *shaderPath,
                       char *shaderProgName)
{
    INFO("Loading vertex shaderProg: " + CATS(shaderPath) +
         ", " + CATS(shaderProgName));

    // Assert the existance of a context
    assert(cgIsContext(cgContext));

    // Get the shader program from the input Cg file
    vertexProg = cgCreateProgramFromFile(cgContext, CG_SOURCE,
        shaderPath, vertexProfile,shaderProgName, NULL);

    // If the shader is not compiled, compile it
    if (!cgIsProgramCompiled(vertexProg))
        cgCompileProgram(vertexProg);

    // Enable the vertex profile, load the shader, then disable the profile
    cgGLEnableProfile(vertexProfile);
    cgGLLoadProgram(vertexProg);
    cgGLDisableProfile(vertexProfile);
}

void LoadFragmentProgram(CGprogram &fragmentProg,
                         char *shaderPath,
                         char *shaderProgName)
{
    INFO("Loading fragment shaderProg: " + CATS(shaderPath) +
         ", " + CATS(shaderProgName));

     // Assert the existance of a context
    assert(cgIsContext(cgContext));

    // Get the shader program from the input Cg file
    fragmentProg = cgCreateProgramFromFile(cgContext, CG_SOURCE,
        shaderPath, fragmentProfile,shaderProgName, NULL);

    // If the shader is not compiled, compile it
    if (!cgIsProgramCompiled(fragmentProg))
        cgCompileProgram(fragmentProg);

    // Enable the vertex profile, load the shader, then disable the profile
    cgGLEnableProfile(fragmentProfile);
    cgGLLoadProgram(fragmentProg);
    cgGLDisableProfile(fragmentProfile);
}

void InitCgContext()
{
    INFO("Initialization of Cg context");

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
            // Reportingand exitting
            INFO("Neither arbvp1 or vp40 vertex profiles supported on this system");
            INFO("EXITTING ... ");
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
            // Reportingand exitting
            INFO("Neither arbfp1 or fp40 fragment profiles supported on this system");
            INFO("EXITING ... ");
            EXIT(0);
        }
    }

    // Load shaders and check for compilation errors
    LoadVertexProgram(vertexProgram,"../Data/RayCastingShader.cg","vertexMain");
    cgErrorCallback();
    LoadFragmentProgram(fragmentProgram,"../Data/RayCastingShader.cg","fragmentMain");
    cgErrorCallback();
}
}

#endif // _CG_H_
