#ifndef _IMAGE_H_
#define _IMAGE_H_


#include <ImageMagick/Magick++.h>
#include <wand/MagickWand.h>

using namespace Magick;
using namespace MagickCore;


struct unsignedcharImage
{
    unsigned char* uCharPixels;
    int size_X;
    int size_Y;
};

struct floatImage
{
    float* floatPixels;
    int size_X;
    int size_Y;
};

namespace Img
{
    void NotmalizeImage(floatImage* fImage);

    floatImage* LoadImageInFloatArray(char* imagePath);

    unsignedcharImage* LoadImageInCharArray(char* imagePath);

    void DisplayImage(Magick::Image* imagePtr);

    unsigned char* ConvertImageToCharArray(Magick::Image* imagePtr);

    Magick::Image* ReadImage(char* imagePath);

    void WriteImage(char* imagePath, Magick::Image* imagePtr);

    void WriteImageFromWand(char* imagePath, MagickWand* wandPtr);
}

#endif // _IMAGE_H_
