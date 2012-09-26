#include "Image.h"

using namespace Magick;
using namespace MagickCore;

Magick::Image* Img::ReadImage(char* imagePath)
{
    MagickWandGenesis();

    MagickWand *mWand = NewMagickWand();

    MagickReadImage(mWand, " Lena ");

    Magick::Image* imagePtr;
    imagePtr = (Magick::Image*) GetImageFromMagickWand(mWand);

    return imagePtr;
}

void Img::DisplayImage()
{

}


void Img::WriteImageFromWand(char* imagePath, MagickWand* wandPtr)
{
    MagickWriteImages(wandPtr, " TestX.png ", MagickTrue);
}


void Img::WriteImage(char* imagePath, Magick::Image* imagePtr)
{
    imagePtr->write("Testxxx.png");
}
