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
#include "Volume/Loader.h"
#include "Volume/Volume.h"
#include "MACROS/MACROS.h"
#include <GL/glut.h>

using namespace std;

// Volume RGBA & luminance images
GLubyte *rgbaImage;
GLubyte *luminanceImage;

// Texture ID
GLuint volumeTexID;

// Display list
GLuint listName;

float xRotation         = 0.0;
float yRotation         = 0.0;
float zRotation         = 0.0;
float scaleFactor       = 0.5;
char rotateState = 0;

// Slice XY half distance
float sliceArmDistance = 0.5;

// Total number of slices
int numSlices = 256;

// Volume dimensions
int iWidth;
int iHeight;
int iDepth;

// Setting the bounding box
int setBoundingBox = 0;

// Orthographic projection parameter
GLfloat presSize = 1.0;

// Transfer function parameters
float rValueTF = 1.0;   // R
float gValueTF = 1.0;   // G
float bValueTF = 1.0;   // B
float aValueTF = 1.0;   // A

// Density threshold value
int desityThresholdTF = 60;

// Total number of voxels in the volume
int numVoxels;

namespace VolumeSlicer
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
        luminanceVal = (rgbaVal < desityThresholdTF) ? 0 : rgbaVal - desityThresholdTF;

        // Division by 2
        luminanceVal = luminanceVal >> 1;

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
    INFO("Volume size: [" + ITS(iWidth) + "X" + ITS(iHeight) + "x" + ITS(iDepth) + "]");

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


void SetDisplayList(void)
{
    // Distance
    float dist;
    int i;

    // Number of half slice
    int nHSlices     = 1.3 * numSlices / 4;
    // int nHSlices = 1.3 * (sqrt(float(iWidth*iWidth + iHeight*iHeight + iDepth*iDepth)))/4.0;

    // Number of slices (including the central one)
    int nSlices      = 2 * nHSlices + 1;

    // Number of lements (verticies)
    int nElements    = nSlices*4;

    // Total number of points
    GLfloat *vPoints    = new GLfloat [3 * nElements];
    GLfloat *ptr        = vPoints;
    float dDist         = (sqrt(3.0) / nSlices);
    float halfDistance  = sliceArmDistance;


    for (i = -nHSlices; i <= nHSlices; i++)
    {
        // Sampling
        dist     = i * dDist;

        *(ptr++) = -halfDistance;
        *(ptr++) = -halfDistance;
        *(ptr++) =  dist;

        *(ptr++) =  halfDistance;
        *(ptr++) = -halfDistance;
        *(ptr++) =  dist;

        *(ptr++) =  halfDistance;
        *(ptr++) =  halfDistance;
        *(ptr++) =  dist;

        *(ptr++) = -halfDistance;
        *(ptr++) =  halfDistance;
        *(ptr++) =  dist;
    }

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, vPoints);
    listName = glGenLists(1);
    glNewList(listName, GL_COMPILE);
    glDrawArrays(GL_QUADS, 0, nElements);
    glEndList();
    delete [] vPoints;
}

GLuint* UploadTextureToGPU(char* volume)
{

}

void Init(char **argv)
{
    if (argv[0] == NULL)
    {
        INFO("NO dataset provided to the Volume Slicer");
        INFO("EXITTING ... ");
        EXIT(0);
    }

    char* datasetPath = argv[1];

    // Read the input volume
    // @TODO
    ReadVolume("/home/abdellah/Software/DataSets/CTData/CTData");

    // Clearing color buffer
    glClearColor (0.0, 0.0, 0.0, 0.0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Generating texture ID and binding it to the GPU
    glGenTextures(1, &volumeTexID);
    glBindTexture(GL_TEXTURE_3D, volumeTexID);

    // Adjusting the texture parameters
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    // For automatic texture coordinate generation
    glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
    glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
    glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);

    // Allocating the texture on the GPU
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA,
                 iWidth, iHeight, iDepth, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, rgbaImage);

    // Enable texturing
    glEnable(GL_TEXTURE_GEN_S);
    glEnable(GL_TEXTURE_GEN_T);
    glEnable(GL_TEXTURE_GEN_R);

    // Setting the blending function & enabling blending
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

    // Setting the display list
    SetDisplayList();
}

void Display(void)
{
    // Enabling 3D texturing
    glEnable(GL_TEXTURE_3D);

    // Update texture and upload it to the GPU
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glBindTexture(GL_TEXTURE_3D, volumeTexID);

    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0,
                    iWidth, iHeight, iDepth, GL_RGBA, GL_UNSIGNED_BYTE, rgbaImage);


    // Cleraing color buffers
    glClear(GL_COLOR_BUFFER_BIT);

    // Clip planes
    static GLdouble eqx0[4] = { 1.0, 0.0, 0.0, 0.0};
    static GLdouble eqx1[4] = {-1.0, 0.0, 0.0, 1.0};
    static GLdouble eqy0[4] = {0.0,  1.0, 0.0, 0.0};
    static GLdouble eqy1[4] = {0.0, -1.0, 0.0, 1.0};
    static GLdouble eqz0[4] = {0.0, 0.0,  1.0, 0.0};
    static GLdouble eqz1[4] = {0.0, 0.0, -1.0, 1.0};

    // Define equations for automatic texture coordinate generation
    static GLfloat x[] = {1.0, 0.0, 0.0, 0.0};
    static GLfloat y[] = {0.0, 1.0, 0.0, 0.0};
    static GLfloat z[] = {0.0, 0.0, 1.0, 0.0};

    // Adjusting the matrix state
    glPushMatrix ();

    // Scaling
    glScalef(scaleFactor, scaleFactor, scaleFactor);

    glPushMatrix ();

    // Transform the viewing direction
    glRotatef(-zRotation, 0.0, 0.0, 1.0);
    glRotatef(-yRotation, 0.0, 1.0, 0.0);
    glRotatef(-xRotation, 1.0, 0.0, 0.0);

    // center the cube
    glTranslatef(-0.5, -0.5, -0.5);

    // Take a copy of the model view matrix now shove it in to the GPU
    // buffer for later use in automatic texture coord generation.
    glTexGenfv(GL_S, GL_EYE_PLANE, x);
    glTexGenfv(GL_T, GL_EYE_PLANE, y);
    glTexGenfv(GL_R, GL_EYE_PLANE, z);

    glClipPlane(GL_CLIP_PLANE0, eqx0);
    glClipPlane(GL_CLIP_PLANE1, eqx1);

    glClipPlane(GL_CLIP_PLANE2, eqy0);
    glClipPlane(GL_CLIP_PLANE3, eqy1);

    glClipPlane(GL_CLIP_PLANE4, eqz0);
    glClipPlane(GL_CLIP_PLANE5, eqz1);

    glPopMatrix ();

    // Enable clip planes
    glEnable(GL_CLIP_PLANE0);
    glEnable(GL_CLIP_PLANE1);
    glEnable(GL_CLIP_PLANE2);
    glEnable(GL_CLIP_PLANE3);
    glEnable(GL_CLIP_PLANE4);
    glEnable(GL_CLIP_PLANE5);

    // Rendering the diaply list of the rectangles
    glCallList(listName);

    glPopMatrix ();

    // Swap the buffers
    glutSwapBuffers();

    // Disabling texturing
    glDisable(GL_TEXTURE_3D);
}


void Idle(void)
{
    // Calculating the FPS
    static clock_t thisClock, startClock = clock();
    static int numTimes = 0;
    int avgCount        = 20;
    float frameDuration;

    if (numTimes == avgCount)
    {
        thisClock = clock();
        frameDuration = (double)(thisClock - startClock) / CLOCKS_PER_SEC;
        startClock = thisClock;
        numTimes = 0;
        COUT << "Frame Rate = " << ((float) avgCount) / frameDuration << ENDL;
    }

    // Redisplay
    Display();

    if (!rotateState)
    {
        xRotation += 1.0;
        yRotation += 1.0;
        zRotation += 1.0;
    }

    numTimes++;
}

void Reshape(int volumeWidth, int volumeHeight)
{
   glViewport(0, 0, (GLsizei) volumeWidth, (GLsizei) volumeHeight);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   presSize = 1.0;
   GLfloat aspect = (GLfloat) volumeHeight/(GLfloat) volumeWidth;
   glOrtho(-presSize, presSize, -presSize * aspect, presSize * aspect, -presSize, presSize);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

void PrintHelp(void)
{
    INFO("See the Keyboard() OpenGL function for more details");
}

void UpdateScene()
{
    UpdateVolume();
    Display();
}

void Keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'q':
            rValueTF -= 0.05;
            INFO("R : " + ITS(rValueTF));
            UpdateScene();
            break;
        case 'Q':
            rValueTF += 0.05;
            INFO("R : " + ITS(rValueTF));
            UpdateScene();
            break;
        case 'w':
            gValueTF -= 0.05;
            INFO("G : " + ITS(gValueTF));
            UpdateScene();
            break;
        case 'W':
            gValueTF += 0.05;
            INFO("G : " + ITS(gValueTF));
            UpdateScene();
            break;
        case 'e':
            bValueTF -= 0.05;
            INFO("B : " + ITS(bValueTF));
            UpdateScene();
            break;
        case 'E':
            bValueTF += 0.05;
            INFO("B : " + ITS(bValueTF));
            UpdateScene();
            break;
        case 'r':
            aValueTF -= 0.05;
            INFO("A : " + ITS(aValueTF));
            UpdateScene();
            break;
        case 'R':
            aValueTF += 0.05;
            INFO("A : " + ITS(aValueTF));
            UpdateScene();
            break;
        case 't':
            desityThresholdTF -= 1;
            INFO("A : " + ITS(desityThresholdTF));
            UpdateScene();
            break;
        case 'T':
            desityThresholdTF += 1;
            INFO("A : " + ITS(desityThresholdTF));
            UpdateScene();
            break;
        case 'a':
            xRotation += 10.0;
            xRotation = (xRotation >= 360.0) ? 0.0 : xRotation;
            INFO("X-axis rotation : " + ITS(xRotation));
            break;
        case 'A':
            xRotation -= 10.0;
            xRotation = (xRotation < 0.0) ? 360.0 : xRotation;
            INFO("X-axis rotation : " + ITS(xRotation));
            break;
        case 'halfDistance':
            yRotation += 10.0;
            yRotation = (yRotation >= 360.0) ? 0.0 : yRotation;
            INFO("Y-axis rotation : " + ITS(yRotation));
            break;
        case 'S':
            yRotation -= 10.0;
            yRotation = (yRotation < 0.0) ? 360.0 : yRotation;
            INFO("Y-axis rotation : " + ITS(yRotation));
            break;
        case 'd':
            zRotation += 10.0;
            zRotation = (zRotation >= 360.0) ? 0.0 : zRotation;
            INFO("Z-axis rotation : " + ITS(zRotation));
            break;
        case 'D':
            zRotation -= 10.0;
            zRotation = (zRotation < 0.0) ? 360.0 : zRotation;
            INFO("Z-axis rotation : " + ITS(zRotation));
            break;
        case 'z':
            scaleFactor *= 0.9;
            INFO("ZoomFactor: " + ITS(scaleFactor));
            break;
        case 'Z':
            scaleFactor *= 1.1;
            INFO("ZoomFactor: " + ITS(scaleFactor));
            break;
        case 'x':
            sliceArmDistance -= 0.05;
            INFO("sliceArmDistance : " + ITS(sliceArmDistance));
            SetDisplayList();
            break;
        case 'X':
            sliceArmDistance += 0.05;
            INFO("sliceArmDistance : " + ITS(sliceArmDistance));
            SetDisplayList();
            break;
        case 'c':
            numSlices -= 1;
            INFO("numSlices : " + ITS(numSlices));
            SetDisplayList();
            break;
        case 'C':
            numSlices += 1;
            INFO("numSlices : " + ITS(numSlices));
            SetDisplayList();
            break;
        case 'O':
            presSize += 0.05;
            Display();
            break;
        case 'o':
            presSize -= 0.05;
            Display();
            break;
        case ' ':
            rotateState = !rotateState;
            break;
        case 'h':
            PrintHelp();
            break;
        case 27:
            delete [] rgbaImage;
            exit(0);
            break;
    }
}

void InitGlut(int argc, char** argv)
{
    INFO("Initializing GLUT");

    // Initialize GLUT context
    glutInit(&argc, argv);

    // Initial window dimensions
    const int initialWinWidth = 512;
    const int initialWinHeight = 512;

    // Calculate window dimensions and center its position
    const int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
    const int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
    const int glutWinPos_X = ((screenWidth - initialWinWidth) / 2);
    const int glutWinPos_Y = ((screenHeight- initialWinHeight) / 2);

    // RGBA and double buffering display mode
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    // Set GLUT window size
    glutInitWindowSize(initialWinWidth, initialWinHeight);

    // Set GLUT window position
    glutInitWindowPosition(glutWinPos_X, glutWinPos_Y);

    // Create GLUT window and set its title
    glutCreateWindow("cuYURI - Volume Slicer");

    INFO("Display Mode : GLUT_RGBA | GLUT_DOUBLE");
    INFO("Initializing GLUT DONE");
}


void RegisterOpenGLCallBacks()
{
    // Display function
    glutDisplayFunc(Display);

    // Reshaping function
    glutReshapeFunc(Reshape);

    // Idle function
    glutIdleFunc(Idle);

    // Keyboard function
    glutKeyboardFunc (Keyboard);
}

void runRenderingEngine(int argc, char** argv)
{
    INFO("Running VOLUME SLICER ... Dataset: " + CATS(argv[1]));

    // GLUT initialization
    InitGlut(argc, argv);

    // Initializing the program and uploading the textures
    Init(argv);

    // Registering OpenGL callbacks
    RegisterOpenGLCallBacks();

    // Run the GLUT main loop
    glutMainLoop();

    return 0;
}
}
