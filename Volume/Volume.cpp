/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File			: Volume
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created		: April 2011
 * Description	:
 * Note(s)		:
 *********************************************************************/

#include "Volume.h"
#include "Utilities/MACROS.h"
#include "Utilities/Utils.h"


int Volume::GetUnifiedDimension(const int iMaxDim)
{
    int eUnifiedDim = 0;

    if      (iMaxDim <= 16)     eUnifiedDim = 16;
    else if (iMaxDim <= 32)     eUnifiedDim = 32;
    else if (iMaxDim <= 64)     eUnifiedDim = 64;
    else if (iMaxDim <= 128)    eUnifiedDim = 128;
    else if (iMaxDim <= 256)    eUnifiedDim = 256;
    else if (iMaxDim <= 512)    eUnifiedDim = 512;
    else if (iMaxDim <= 1024)   eUnifiedDim = 1024;
    else if (iMaxDim <= 2048)   eUnifiedDim = 2048;
    else if (iMaxDim <= 4096)   eUnifiedDim = 4096;
    else if (iMaxDim <= 8192)   eUnifiedDim = 8192;
    else if (iMaxDim <= 16384)  eUnifiedDim = 16384;

    return eUnifiedDim;
}
