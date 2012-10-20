#include "Image.h"
#include "iostream"
#include "Utilities/MACROS.h"

using namespace Magick;
using namespace MagickCore;

Magick::Image* Img::ReadImage(char* imagePath)
{
    MagickWandGenesis();

    MagickWand *mWand = NewMagickWand();

    INFO("Readiung the image");
    if(MagickReadImage(mWand, "Lena")
            == MagickFalse)
    {
        INFO("Can't READ the image");
    }
    else
        INFO("Image has been read SUCCESSFULLY");

    // Writing with the wand
    MagickWriteImage(mWand,"LenaTest.jpg");


    INFO("Getting the image from the WAND");
    Magick::Image* imagePtr;
    imagePtr = (Magick::Image*) GetImageFromMagickWand(mWand);



     Magick::Image imageRef;
     imageRef.read("Lena");
     imageRef.crop( Geometry(100,100, 100, 100) );
     imageRef.write( "x.gif" );


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
    INFO("TEst");
    //unsigned char *pixels = new unsigned char[imagePtr->size().width() * imagePtr->size().height()];
    //image.size().height();
    //image.write( 0, 0, imagePtr->size().width(), imagePtr->size().height(), "I", CharPixel, pixels );
    std::cout << imagePtr->size().width();
 INFO("DONETEST")
    imagePtr->write("Testxxx.png");
   ;
}
