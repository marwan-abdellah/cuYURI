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

#ifndef _LOADING_VOLUME_H_
#define _LOADING_VOLUME_H_

#include "VolumeRayCaster.h"

#include "Cg.hpp"
#include "ColorCube.hpp"
#include "GL_CallBacks.hpp"
#include "GL_Buffers.hpp"
#include "GLEW.hpp"
#include "GLUT.hpp"
#include "LoadingVolume.hpp"
#include "Rendering.hpp"
#include "VolumeData.hpp"

namespace RayCaster
{
void ReadHeader(char *prefix,
                int &volumeWidth, int &volumeHeight, int &volumeDepth)
{
    char hdrFile[300];
    std::ifstream inputFileStream;

    // Adding the ".hdr" prefix to the dataset path
    sprintf(hdrFile, "%s.hdr", prefix);

    INFO("Dataset HDR hdrFile : " + CATS(hdrFile));

    // Open the eader hdrFile to read the dataset dimensions
    inputFileStream.open(hdrFile, std::ios::in);

    // Checking the openning of the file
    if (inputFileStream.fail())
    {
        INFO("Could not open the HDR file :" + CATS(hdrFile));
        INFO("Exiting");
        EXIT(0);
    }

    // Reading the dimensions
    inputFileStream >> volumeWidth;
    inputFileStream >> volumeHeight;
    inputFileStream >> volumeDepth;

    // Closing the ".hdr" file
    inputFileStream.close();

    INFO("HDR file has been read SUCCESSFULLY");
}

void UpdateVolume()
{
    // Poiter to the volume image
    GLubyte *ptr = luminanceImage;

    if (setBoundingBox)
    {
        // Put a box around the volume so that we can see the outline
        // of the data.
        INFO("Setting the Bounding Box");

        int i, j, k;
        for (i = 0; i < iDepth; i++) {
            for (j = 0; j < iHeight; j++) {
                for (k = 0; k < iWidth; k++) {
                    if (((i < 4) && (j < 4)) ||
                        ((j < 4) && (k < 4)) ||
                        ((k < 4) && (i < 4)) ||
                        ((i < 4) && (j >  iHeight-5)) ||
                        ((j < 4) && (k > iWidth-5)) ||
                        ((k < 4) && (i > iDepth-5)) ||
                        ((i > iDepth-5) && (j >  iHeight-5)) ||
                        ((j >  iHeight-5) && (k > iWidth-5)) ||
                        ((k > iWidth-5) && (i > iDepth-5)) ||
                        ((i > iDepth-5) && (j < 4)) ||
                        ((j >  iHeight-5) && (k < 4)) ||
                        ((k > iWidth-5) && (i < 4))) {
                        *ptr = 110;
                    }
                    ptr++;
                }
            }
        }
    }
    else
        INFO("NO bounding box");

    ptr = luminanceImage;

    // Pointer to the RGBA image
    GLubyte *qtr = rgbaImage;

    // Luminance & RGBA values
    GLubyte rgbaVal, luminanceVal;

    // Reading the luminance volume and constructing the RGBA volume
    for (int i = 0; i < numVoxels; i++)
    {
        rgbaVal = *(ptr++);

        // Area of interest
        if (rgbaVal > desityThresholdTF)
            luminanceVal = 0;
        else
            luminanceVal = rgbaVal * 2;

        *(qtr++) = ((float)luminanceVal) * rValueTF;
        *(qtr++) = ((float)luminanceVal) * gValueTF;
        *(qtr++) = ((float)luminanceVal) * bValueTF;
        *(qtr++) = ((float)luminanceVal) * aValueTF;
    }
}

void ReadVolume(char *prefix)
{
    char imgFile[100];
    ifstream inputFileStream;

    // Reading the header file
    ReadHeader(prefix, iWidth, iHeight, iDepth);
    INFO("Volume size: [" + ITS(iWidth) + "X" +
         ITS(iHeight) + "x" + ITS(iDepth) + "]");

    // Adding the ".img" prefix to the dataset path
    sprintf(imgFile, "%s.img", prefix);
    INFO("Reading the volume file " + CATS(imgFile));

    // Total number of voxels
    numVoxels = iWidth * iHeight * iDepth;
    INFO("Number of voxels : " + ITS(numVoxels));

    // Allocating the luminance image
    luminanceImage = new GLubyte [numVoxels];

    // Allocating the RGBA image
    rgbaImage = new GLubyte [numVoxels * 4];

    // Reading the volume image (luminance values)
    inputFileStream.open(imgFile, ios::in);
    if (inputFileStream.fail())
    {
        INFO("Could not open " + CATS(imgFile));
        EXIT(0);
    }

    // Read the image byte by byte
    inputFileStream.read((char *)luminanceImage, numVoxels);

    // Closing the input volume stream
    inputFileStream.close();

    // Update the volume
    UpdateVolume();

    INFO("The volume has been read successfull and the RGBA one is DONE");
}
}
#endif // _LOADING_VOLUME_H_
