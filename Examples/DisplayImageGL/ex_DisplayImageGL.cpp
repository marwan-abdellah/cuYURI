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

#include "ex_DisplayImageGL.h"
#include "CUDA/cuGlobals.h"
#include "CUDA/Utilities/cuUtilities.h"
#include "CUDA/cuYURI_Interfaces.h"
#include "Globals.h"
#include "Utilities/Utils.h"
#include "MACROS/MACROS.h"
#include "MACROS/MemoryMACROS.h"
#include "Dependencies/Excel/ExcelFormat.h"
#include <ImageMagick/Magick++.h>

#include "OpenGL/OpenGL.h"
#include "Image/Image.h"
using namespace ExcelFormat;

void ex_DisplayImageGL::run(int argc, char** argv)
{
    // Reading the image
    INFO("Reading a sample image");
    Magick::Image* imgPtr = Img::ReadImage("../Data/Lena");

    // Image dimensions
    const int NX = imgPtr->size().width();
    const int NY = imgPtr->size().height();
    const int NXY = NX * NY;

    // Convert the image to a char array to be displayed with OpenGL
    INFO("Converting the image to unsigned char array");
    unsigned char* uCharPixels;
    uCharPixels = Img::ConvertImageToCharArray(imgPtr);

    // Converting the image from char array to float
    // array for high precision
    float* floatPixels = MEM_ALLOC_1D_GENERIC(float, NXY);
    for (int i = 0; i < NXY; i++)
        floatPixels[i] = (float) uCharPixels[i];

    // Normalizing the array between 0 & 1
    int maxValue = 0;
    for (int i = 0; i < NXY; i++)
    {
        if (floatPixels[i] > maxValue)
            maxValue = floatPixels[i];
    }

    for (int i = 0; i < NXY; i++)
        floatPixels[i] /= maxValue;

    // Freeing uCharPixels
    FREE_MEM_1D(uCharPixels, unsigned char);

    // Freeing uCharPixels
    delete imgPtr;

    // Directly display an image on a 2D OpenGL widget
    INFO("Display the image n OpenGL widget")
    OpenGL::DisplayImage(floatPixels, NX, NY);

    INFO("DONE");
}
