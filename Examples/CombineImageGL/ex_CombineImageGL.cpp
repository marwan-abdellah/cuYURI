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

#include "ex_CombineImageGL.h"
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

void ex_CombineImageGL::run(int argc, char** argv)
{
    // Reading the image
    INFO("Reading sample images");
    Magick::Image* imgPtr_1 = Img::ReadImage("../Data/Lena");
    Magick::Image* imgPtr_2 = Img::ReadImage("../Data/Lady");

    // Image dimensions
    const int NX_1 = imgPtr_1->size().width();
    const int NY_1 = imgPtr_1->size().height();

    const int NX_2 = imgPtr_2->size().width();
    const int NY_2 = imgPtr_2->size().height();

    if (NX_1 == NX_2 && NY_1 == NY_2)
    {
        INFO("Image dimensions match");
    }
    else
    {
        INFO("Image dimensions don't match. CAN'T PROCEED !");
        EXIT(0);
    }

    const int NX = NX_1;
    const int NY = NY_1;
    const int NXY = NX * NY;

    // Writing the images to the disk with a different name
    INFO("Writing the images");
    Img::WriteImage("../Data/LenaModified", imgPtr_1);
    Img::WriteImage("../Data/LadyModified", imgPtr_2);

    // Convert the image to a char array to be displayed with OpenGL
    INFO("Converting the images to unsigned char array");
    unsigned char* uCharPixels_1;
    unsigned char* uCharPixels_2;
    uCharPixels_1 = Img::ConvertImageToCharArray(imgPtr_1);
    uCharPixels_2 = Img::ConvertImageToCharArray(imgPtr_2);

    // Converting the image from char array to float
    // array for high precision
    float* floatPixels_1 = MEM_ALLOC_1D_GENERIC(float, NXY);
    float* floatPixels_2 = MEM_ALLOC_1D_GENERIC(float, NXY);

    float* floatPixels_Results = MEM_ALLOC_1D_GENERIC(float, NXY);
    for (int i = 0; i < NXY; i++)
    {
        floatPixels_1[i] = (float) uCharPixels_1[i];
        floatPixels_2[i] = (float) uCharPixels_2[i];
    }

    // Normalizing the arrays between 0 & 1
    int maxValue_1 = 0;
    int maxValue_2 = 0;
    for (int i = 0; i < NXY; i++)
    {
        if (floatPixels_1[i] > maxValue_1)
            maxValue_1 = floatPixels_1[i];

        if (floatPixels_2[i] > maxValue_2)
            maxValue_2 = floatPixels_2[i];
    }

    for (int i = 0; i < NXY; i++)
    {
        floatPixels_1[i] /= maxValue_1;
        floatPixels_2[i] /= maxValue_2;
    }

    // Adding the 2 images in 1 image
    for (int i = 0; i < NXY; i++)
        floatPixels_Results[i] = floatPixels_1[i] + floatPixels_2[i];

    // Normalizing
    for (int i = 0; i < NXY; i++)
        floatPixels_Results[i] /= 2;

    // Freeing uCharPixels
    FREE_MEM_1D(uCharPixels_1, unsigned char);
    FREE_MEM_1D(uCharPixels_2, unsigned char);

    // Freeing uCharPixels
    delete imgPtr_1;
    delete imgPtr_2;

    // Freeing floatPixels
    FREE_MEM_1D(floatPixels_1, float);
    FREE_MEM_1D(floatPixels_2, float);

    // Directly display an image on a 2D OpenGL widget
    INFO("Display the image n OpenGL widget")
    OpenGL::DisplayImage(floatPixels_Results, NX, NY);

    FREE_MEM_1D(floatPixels_Results, float);

    INFO("DONE");
}
