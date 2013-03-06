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

#include "ex_GPUCombineDisplayImageGL.h"
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

// Sets the kernel GPU configuration automatically
#define AUTO_CONF 1

void ex_GPUCombineDisplayImageGL::run(int argc, char** argv)
{
    // Reading the image
    INFO("Adding images on GPU and display the result on OpenGL widget");

    // Load images into float arrays
    floatImage* fImage_1 = Img::LoadImageInFloatArray("../Data/Lena");
    floatImage* fImage_2 = Img::LoadImageInFloatArray("../Data/Lady");

    if (fImage_1->size_X == fImage_2->size_X &&
            fImage_1->size_Y == fImage_2->size_Y)
    {
        INFO("Images are with Unified dimensions");
    }
    else
    {
        INFO("Images don't have unified dimensions ... EXITTING");
        EXIT(0);
    }

    // Normalize images
    Img::NotmalizeImage(fImage_1);
    Img::NotmalizeImage(fImage_2);

    // Host (CPU) image
    float *fImageOutput;

    // Device (GPU) images
    float *dev_fImage_1;
    float *dev_fImage_2;
    float *dev_fImageOutput;

    // Image dimensions must be the same
    const int NX = fImage_1->size_X;
    const int NY = fImage_1->size_Y;
    const int NXY = NX * NY;

    // Allocating the host array
    INFO("Allocating HOST arrays");
    fImageOutput = MEM_ALLOC_1D(float, NXY);

    // Allocating device arrays
    INFO("Allocating device arrays");
    dev_fImage_1 = cuUtils::Create_Device_Vector <float> (NXY);
    dev_fImage_2 = cuUtils::Create_Device_Vector <float> (NXY);
    dev_fImageOutput = cuUtils::Create_Device_Vector <float> (NXY);

     // Upload images to the GPU
    INFO("Iploading images to GPU");
    cuUtils::Upload_1D_Array <float> (fImage_1->floatPixels, dev_fImage_1, NXY);
    cuUtils::Upload_1D_Array <float> (fImage_2->floatPixels, dev_fImage_2, NXY);

    // GPU profiler
    cuProfile profile;

#ifdef AUTO_CONF
    // Generate optimized AUTOMATIC kernel configuration for the GPU
    kernelConf* autoConf = cuUtils::AutoGenKernelConf_1D(NXY);
#else
    // Kernel configuration
    // NOTE: N should be power-of-two to have a SUCCESSFUL kernel execution
    // for the manual configuration
    dim3 cuBlock(2, 1, 1);
    dim3 cuGrid(NXY / cuBlock.x, 1, 1);
#endif

    // Launch the kernel and get statistics
    INFO("Executing image addition kernel");
    cuYURI::cu_Add_1D_Arrays <float>
            (autoConf->cuBlock, autoConf->cuGrid,
             dev_fImage_1, dev_fImage_2, dev_fImageOutput,
             NXY, &profile);

    // Display profiling data
    INFO("Profiling data")
    cuUtils::DisplayKernelProfilingData("@cu_Add_1D_Arrays", &profile);

    // Download the final image to the host side
    INFO("Downloading resulting image to the HOST side")
    cuUtils::Download_1D_Array <float>
            (fImageOutput, dev_fImageOutput, NXY);

    // Display image on OpenGL widget
    INFO("Display the image on OpenGL widget")
    OpenGL::DisplayImage(fImageOutput, NX, NY);

    INFO("DONE");
}
