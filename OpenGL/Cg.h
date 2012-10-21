#ifndef _CG_H_
#define _CG_H_

#include <Cg/cg.h>
#include <Cg/cgGL.h>
#include "MACROS/MACROS.h"

namespace OpenGL
{
namespace Cg
{

void cgErrorCallback(CGcontext currentContext);

void SetTextureUniformParams(char* paramName, GLuint iTex_ID,
                             const CGprogram &shaderProg,
                             CGparameter paramVal);

void LoadVertexProgram(CGcontext currentContext,
                       CGprofile vertexProfile,
                       CGprogram &vertexProg,
                       char *shaderFilePath, char *progName);

void LoadFragmentProgram(CGcontext currentContext,
                         CGprofile fragmentProfile,
                         CGprogram &fragmentProg,
                         char *shaderFilePath, char *progName);

CGprofile GetVertexProfile(CGcontext currentContext);
CGprofile GetFragmentProfile(CGcontext currentContext);


}
}
#endif // _CG_H_
