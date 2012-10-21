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

#ifndef _CG_H_
#define _CG_H_

#include "VolumeRayCaster.h"

#include "Cg.hpp"
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
        INFO (cgGetErrorString(cgError));

        if(cgContext != NULL)
            INFO(CATS("Last Cg cgContext : ") + cgGetLastListing(cgContext));

        // Exiting application
        EXIT(0);
    }
}

// Sets a uniform texture parameter
void SetUnformTexParameter(char* par, GLuint tex,
                           const CGprogram &program,
                           CGparameter param)
{
    param = cgGetNamedParameter(program, par);
    cgGLSetTextureParameter(param, tex);
    cgGLEnableTextureParameter(param);
}


// LoadVertexProgram: loading a vertex program
void LoadVertexProgram(CGprogram &v_program,
                       char *shader_path,
                       char *program_name)
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
void LoadFragmentProgram(CGprogram &f_program,
                         char *shader_path,
                         char *program_name)
{

    assert(cgIsContext(cgContext));
    f_program = cgCreateProgramFromFile(cgContext, CG_SOURCE,
                                        shader_path,
        fragmentProfile,program_name, NULL);
    if (!cgIsProgramCompiled(f_program))
        cgCompileProgram(f_program);

    cgGLEnableProfile(fragmentProfile);
    cgGLLoadProgram(f_program);
    cgGLDisableProfile(fragmentProfile);
}

void InitCgContext()
{
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
    LoadVertexProgram(vertexProgram,"../Data/RayCastingShader.cg","vertexMain");
    cgErrorCallback();
    LoadFragmentProgram(fragmentProgram,"../Data/RayCastingShader.cg","fragmentMain");
    cgErrorCallback();
}

}


#endif // _CG_H_
