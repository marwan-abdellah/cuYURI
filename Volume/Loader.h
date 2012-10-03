#ifndef _LOADER_H_
#define _LOADER_H_

#include "Globals.h"

namespace Volume
{
volumeSize* LoadHeader(const char *path);
volumeImage* LoadVolume(const char* path);

}

#endif // _LOADER_H_
