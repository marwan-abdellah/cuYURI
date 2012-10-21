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

#include "ex_SimpleDisplayImageGL.h"
#include "CUDA/cuGlobals.h"
#include "CUDA/Utilities/cuUtilities.h"
#include "CUDA/cuYURI_Interfaces.h"
#include "Globals.h"
#include "Utilities/Utils.h"
#include "MACROS/MACROS.h"
#include "MACROS/MemoryMACROS.h"
#include <ImageMagick/Magick++.h>

#include "OpenGL/OpenGL.h"
#include "Image/Image.h"

void ex_SimpleDisplayImageGL::run(int argc, char** argv)
{
    // Reading the image
    INFO("Direct display of a simple image on OpenGL widget");

    // Load the image into float array
    floatImage* fImage = Img::LoadImageInFloatArray("../Data/Lena");

    // Normalize image
    Img::NotmalizeImage(fImage);

    // Display image on OpenGL widget
    INFO("Display the image on OpenGL widget")
    OpenGL::DisplayImage(fImage->floatPixels, fImage->size_X, fImage->size_Y);

    INFO("DONE");
}
