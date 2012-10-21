#include "Image.h"
#include "iostream"
#include "Utilities/Utils.h"
#include "MACROS/MACROS.h"
#include "MACROS/MemoryMACROS.h"

using namespace Magick;
using namespace MagickCore;

Magick::Image* Img::ReadImage(char* imagePath)
{
    Magick::Image* imagePtr = new Magick::Image();

    // Read the image
    INFO("Reading image");
    imagePtr->read(imagePath);

    INFO("Image dimensions : "
         + CATS("[") + ITS(imagePtr->size().width()) + CATS("]") + " X "
         + CATS("[") + ITS(imagePtr->size().height()) + CATS("]"));

    return imagePtr;
}

void Img::WriteImage(char* imagePath, Magick::Image* imagePtr)
{
    INFO("Writing image to disk");

    imagePtr->write(imagePath);
}

unsigned char* Img::ConvertImageToCharArray(Magick::Image* imagePtr)
{
    INFO("Converting the Magick image to unsigned char image");

    // Image size in bytes
    const int imageSizeBytes = imagePtr->size().width() * imagePtr->size().height();

    // Unsigned char image
    unsigned char* uCharPixels = MEM_ALLOC_1D_GENERIC(unsigned char, imageSizeBytes);

    // Write image data into the char array
    imagePtr->write(0, 0, imagePtr->size().width(), imagePtr->size().height(),
                    "I", CharPixel, uCharPixels);

    return uCharPixels;
}

unsignedcharImage* Img::LoadImageInCharArray(char* imagePath)
{
    INFO("Loading image into unsigned char array")
    // Load the image in Magick container
    Magick::Image* imagePtr;
    imagePtr = Img::ReadImage(imagePath);

    // uChar image
    unsigned char* uCharPixels;
    uCharPixels = Img::ConvertImageToCharArray(imagePtr);

    // Image size
    const int NX = imagePtr->size().width();
    const int NY = imagePtr->size().height();
    const int NXY = NX * NY;

    unsignedcharImage* uCharImage =
            MEM_ALLOC_1D_GENERIC(unsignedcharImage, NXY);

    uCharImage->uCharPixels = uCharPixels;
    uCharImage->size_X = NX;
    uCharImage->size_Y = NY;

    // Free imagePtr
    delete imagePtr;

    return uCharImage;
}

floatImage* Img::LoadImageInFloatArray(char* imagePath)
{
    INFO("Loading image into float array");

    // Loading the image in char array
    unsignedcharImage* uCharImage =
            LoadImageInCharArray(imagePath);

    // Float image structure
    floatImage* fImage = MEM_ALLOC_1D_GENERIC(floatImage, 1);
    fImage->size_X = uCharImage->size_X;
    fImage->size_Y = uCharImage->size_Y;
    fImage->floatPixels =
            MEM_ALLOC_1D_GENERIC(float, fImage->size_X * fImage->size_Y);

    // Convert the unsigned char pixels to float pixels
    const int NXY = fImage->size_X * fImage->size_Y;
    for (int i = 0; i < NXY; i++)
        fImage->floatPixels[i] = (float) uCharImage->uCharPixels[i];

    // Freeing memory
    FREE_MEM_1D(uCharImage->uCharPixels, unsigned char);

    return fImage;
}

void Img::NotmalizeImage(floatImage* fImage)
{
    int maxValue = 0;

    // Image length
    const int NXY = fImage->size_X * fImage->size_Y;

    // Get the maximum value
    for (int i = 0; i < NXY; i++)
    {
        if (fImage->floatPixels[i] > maxValue)
            maxValue = fImage->floatPixels[i];
    }

    // Do the normalization
    for (int i = 0; i < NXY; i++)
        fImage->floatPixels[i] /= maxValue;
}


