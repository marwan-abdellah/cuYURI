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

#include "ex_DisplayArrayGL.h"
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

using namespace ExcelFormat;

// Select the image
#define IMAGE 0

void ex_DisplayArrayGL::run(int argc, char** argv)
{
    INFO("Displaying a sample float array on OpenGL Widget");

    // Array dimensions
    int NX = 256;       // X
    int NY = 256;       // Y
    int NXY = NX * NY;  // Size = XY

    INFO("Creating sample image of size "
         + CATS("[") + ITS(NX) + CATS("]") + "X"
         + CATS("[") + ITS(NY) + CATS("]"));

    // Allocate the array
    float* sampleFloatArray = MEM_ALLOC_1D(float, NXY);

    // Fill the array with shades of gray
    int index = 0;

    if (IMAGE)
    {
        for (int i = 0; i < NX; i++)
        {
            for (int j = 0; j < NY; j++)
            {
                if (i < NX / 2 && j < NY / 2)
                    sampleFloatArray[index] = 0.25;
                else if (i > NX / 2 && j < NY / 2)
                    sampleFloatArray[index] = 0.5;
                else if (i < NX / 2 && j > NY / 2)
                    sampleFloatArray[index] = 0.75;
                else
                    sampleFloatArray[index] = 1;

                index ++;
            }
        }
    }
    else
    {
        // Filling the array
        for (int i = 0; i < NXY; i++)
            sampleFloatArray[i] = i;

        // Normalizing the array between 0 & 1
        int maxValue = 0;
        for (int i = 0; i < NXY; i++)
        {
            if (sampleFloatArray[i] > maxValue)
                maxValue = sampleFloatArray[i];
        }

        for (int i = 0; i < NXY; i++)
            sampleFloatArray[i] /= maxValue;

    }

    // Directly display an image on a 2D OpenGL widget
    OpenGL::DisplayImage(sampleFloatArray, NX, NY);

    // Freeing memory
    FREE_MEM_1D(sampleFloatArray, float);

    INFO("DONE");
}
