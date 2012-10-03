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

#ifndef _VOLUME_H_
#define _VOLUME_H_


#include <X11/X.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <math.h>
#include <sys/timeb.h>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "Globals.h"
#include "Utilities/Utils.h"


namespace Volume
{
int GetUnifiedDimension(const int iMaxDim);
volumeImage* CreateTestVolume(const int N);
}

#endif /* VOLUME_H_ */
