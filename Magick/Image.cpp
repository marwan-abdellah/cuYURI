#include "Image.h"
#include "iostream"
#include "Utilities/MACROS.h"

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

void Img::DisplayImage(Magick::Image* imagePtr)
{

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

    imagePtr->write(0, 0, imagePtr->size().width(), imagePtr->size().height(),
                    "I", CharPixel, uCharPixels);

    return uCharPixels;
}


