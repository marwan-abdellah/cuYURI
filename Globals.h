/*
 * Globals.h
 *
 *  Created on: May 19, 2012
 *      Author: abdellah
 */

#ifndef _GLOBALS_H_
#define _GLOBALS_H_

#include <fftw3.h>

/*
 * Gets rid of the annoying warrning
 * "deprecated conversion from string constant to ‘char*’ [-Wwrite-strings]"
 */
#pragma GCC diagnostic ignored "-Wwrite-strings"

/* @ Max float for the rand() function */
#define FLOAT_MAX 214748364

/* @ Max GPU  Memory in MB */
#define MAX_GPU_MEMORY_MB 256
#define MAX_GPU_MEMORY MAX_GPU_MEMORY_MB * 1024 * 1024

struct volumeDimensions
{
    int size_X;
    int size_Y;
    int size_Z;
};

struct volumeSize
{
    int NX;     // X
    int NY;     // Y
    int NZ;     // Z
    int NU;     // Unified Dimension (X = Y = Z)
};


struct volumeImage
{
    // Volume dimensiosn
    volumeSize volSize;

    // Pointer to the volume image data
    char* volPtr;
};

#endif /* _GLOBALS_H_ */
