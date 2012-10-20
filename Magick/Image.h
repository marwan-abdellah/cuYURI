#ifndef _IMAGE_H_
#define _IMAGE_H_


#include <ImageMagick/Magick++.h>
#include <wand/MagickWand.h>

using namespace Magick;
using namespace MagickCore;

namespace Img
{
    void DisplayImage(Magick::Image* imagePtr);

    unsigned char* ConvertImageToCharArray(Magick::Image* imagePtr);

    Magick::Image* ReadImage(char* imagePath);

    void WriteImage(char* imagePath, Magick::Image* imagePtr);

    void WriteImageFromWand(char* imagePath, MagickWand* wandPtr);
}

#endif // _IMAGE_H_
