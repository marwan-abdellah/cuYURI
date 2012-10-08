/**************************************************************************
*        FILE NAME:  VolumeRenderPyramid.cpp
*
* ONE LINE SUMMARY:
*        Tested on NVIDIA Quadro FX3000
* DESCRIPTION:
*        Demonstrates use of 3D textures for volume rendering
*        ultrasound data acquired in pyramidal grids.
*
*    Thilaka Sumanaweera
*    (C) 2002 Siemens Medical Solutions USA, Inc, Ultrasound Division
*    Thilaka.Sumanaweera@siemens.com
*    05/16/2003
*
*****************************************************************************
* REVISION HISTORY:
* Rev     	When      	Who         	What
* A	  	 16May03        sumane		    Created.
****************************************************************************/
#include <fstream.h>
#include <iostream.h>
#include "glut.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <windows.h>
#include "glext.h"
#include <fstream.h>
#include <stdlib.h>
#include <stdio.h>
#include <Cg/cg.h>
#include <Cg/cgGL.h>

// Transfer and Opacity Functions
//
unsigned char RedTransferFunction[256]   = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 21, 30, 36, 42, 49, 60, 64, 68, 73, 77, 81, 89, 94, 96, 100, 103, 107, 113, 116, 120, 122, 125, 127, 132, 135, 137, 140, 142, 145, 146, 151, 153, 155, 158, 159, 162, 165, 167, 169, 170, 173, 175, 176, 180, 181, 182, 184, 186, 187, 189, 191, 193, 195, 196, 198, 199, 200, 201, 204, 205, 207, 207, 209, 211, 211, 214, 215, 216, 217, 219, 219, 220, 222, 223, 224, 226, 226, 227, 228, 230, 231, 232, 232, 234, 235, 236, 237, 238, 239, 239, 241, 242, 243, 243, 245, 245, 246, 246, 243, 242, 240, 239, 238, 237, 235, 234, 233, 232, 232, 231, 230, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229, 229};
unsigned char GreenTransferFunction[256] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 7, 9, 11, 14, 16, 18, 20, 21, 23, 27, 31, 31, 34, 36, 39, 44, 45, 48, 50, 53, 55, 57, 62, 64, 66, 69, 72, 74, 79, 82, 83, 86, 88, 92, 94, 99, 102, 104, 107, 109, 111, 114, 119, 122, 124, 128, 129, 132, 134, 138, 142, 145, 148, 150, 152, 155, 158, 163, 165, 168, 170, 173, 176, 179, 180, 186, 189, 191, 194, 195, 199, 204, 206, 209, 211, 214, 216, 221, 224, 227, 230, 232, 235, 238, 242, 245, 247, 250, 253, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
unsigned char BlueTransferFunction[256]  = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 29, 38, 48, 55, 63, 69, 82, 90, 95, 100, 105, 111, 120, 125, 130, 134, 135, 134, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135, 135};
unsigned char OpacityFunction[256]      = {0, 2, 4, 7, 9, 11, 14, 16, 19, 21, 23, 26, 28, 30, 33, 35, 38, 40, 42, 45, 47, 50, 52, 54, 57, 59, 61, 64, 66, 69, 71, 73, 76, 78, 81, 83, 85, 88, 90, 92, 95, 97, 100, 102, 104, 107, 109, 112, 114, 116, 119, 121, 123, 126, 128, 131, 133, 135, 138, 140, 142, 145, 147, 150, 152, 154, 157, 159, 162, 164, 166, 169, 171, 173, 176, 178, 181, 183, 185, 188, 190, 193, 195, 197, 200, 202, 204, 207, 209, 212, 214, 216, 219, 221, 224, 226, 228, 231, 233, 235, 238, 240, 243, 245, 247, 250, 252, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};

// Cg Context
static CGcontext Context         = NULL;

// Vertex Program Stuff
static CGprogram vProgram        = NULL;
static CGparameter ZoomFactor    = NULL;
static CGparameter Pyramid       = NULL;
static CGparameter ClipPlane0    = NULL;
static CGparameter ClipPlane1    = NULL;
static CGparameter ClipPlane2    = NULL;
static CGparameter ClipPlane3    = NULL;
static CGparameter ClipPlane4    = NULL;
static CGparameter ClipPlane5    = NULL;
static CGparameter ModelView     = NULL;
static CGparameter ModelViewProj = NULL;
static CGprofile vProfile;

// Fragment Program Stuff
static CGprogram fProgram        = NULL;
static CGparameter USTexture     = NULL;
static CGparameter ColorMap      = NULL;
static CGprofile fProfile;

GLuint GL_ColorMap, GL_USTexture, GL_DisplayList;
int ColorMapLength = 256;

GLubyte *image, *ColorMapArray;
int iWidth;
int iHeight;
int iDepth;
int nVolumes;
int VolSize;

float alpha = 340.0;
float theta = 200.0;
float gamma =  15.0;
float scale =   0.7;
char step_p = 1;

const GLfloat M_PI = 3.1415926535897932384626433832795029;
PFNGLTEXIMAGE3DEXTPROC glTexImage3DEXT = NULL;
PFNGLTEXSUBIMAGE3DEXTPROC glTexSubImage3DEXT = NULL;

float Aperture    = 0.2;
float ApexAngle_2 = 15.0;

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   d2r
*
* DESCRIPTION:
*
*   This converts degrees to radians
*
* FORMAL PARAMETERS:
*   angle is in degrees
*
* RETURNS:
*   angle in radians
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
float d2r(float angle)
{
	return(angle*M_PI/180.0);
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   ReadHeader
*
* DESCRIPTION:
*
*   This reads in the header of the 3D data volume
*
* FORMAL PARAMETERS:
*   prefix contains the file prefix.
*
* RETURNS:
*   w, h, and d are width, height and depth of the volume
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void ReadHeader(char *prefix, int &w, int &h, int &d)
{
	char file[300];
	ifstream ifile;
	char str[200];

	sprintf(file, "%s.hdr", prefix);
	cerr << file << ": ";
	ifile.open(file, ios::in | ios::nocreate);

	if (ifile.fail()) {
		cerr << "Could not open " << file << endl;
		exit(0);
	}

	ifile >> w;
	ifile >> h;
	ifile >> d;
	ifile.close();
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   ReadVolume
*
* DESCRIPTION:
*
*   This reads in the 3D data volume
*
* FORMAL PARAMETERS:
*   prefix contains the file prefix.
*
* RETURNS:
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void ReadVolume(char *dir, char *fileprefix, unsigned char *ptr)
{
	char file[100];
	char prefix[200];
	ifstream ifile;

	sprintf(prefix, "%s%s", dir, fileprefix);
	ReadHeader(prefix, iWidth, iHeight, iDepth);
	cerr << "[" << iWidth << "x" << iHeight << "x" << iDepth << "]" << endl;
	
	sprintf(file, "%s.img", prefix);
	int Size = iWidth*iHeight*iDepth;

	ifile.open(file, ios::in | ios::nocreate);
	if (ifile.fail()) {
		cerr << "Could not open " << file << endl;
		exit(0);
	}
	ifile.setmode(filebuf::binary);
	ifile.read((signed char *)ptr, Size);
	ifile.close();

	// Put a box around the volume so that we can see the outline 
	// of the data.
	//
	int i, j, k;
    for (i = 0; i < iDepth; i++) {
        for (j = 0; j < iHeight; j++) {
            for (k = 0; k < iWidth; k++) {
				if (((i < 2) && (j < 2)) ||
					((j < 2) && (k < 2)) ||
					((k < 2) && (i < 2)) ||
					((i < 2) && (j >  iHeight-3)) ||
					((j < 2) && (k > iWidth-3)) ||
					((k < 2) && (i > iDepth-3)) ||
					((i > iDepth-3) && (j >  iHeight-3)) ||
					((j >  iHeight-3) && (k > iWidth-3)) ||
					((k > iWidth-3) && (i > iDepth-3)) ||
					((i > iDepth-3) && (j < 2)) ||
					((j >  iHeight-3) && (k < 2)) ||
					((k > iWidth-3) && (i < 2))) {
					*ptr = 40;
				}
				else {
					// Divide by two to prevent sturation of 8bit channels
					*ptr = *ptr >> 1;
				}
				ptr++;
			}
		}
	}
}


/*************************[MAN-BEG]*******************************************
*
* NAME:
*   ReadVolume
*
* DESCRIPTION:
*
*   This reads in the 3D data volume
*
* FORMAL PARAMETERS:
*   prefix contains the file prefix.
*
* RETURNS:
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void ReadVolumes(char *dir, char *file)
{
	const int MaxLine = 200;
	char vfile[MaxLine];
	char path[MaxLine];
	char prefix[MaxLine];
	ifstream ifile;

	sprintf(path, "%s/%s", dir, file);
	ifile.open(path, ios::in);
	ifile >> nVolumes;
	ifile.getline(vfile, MaxLine, '\n');
	ifile.getline(vfile, MaxLine, '\n');
	sprintf(prefix, "%s%s", dir, vfile);
	ReadHeader(prefix, iWidth, iHeight, iDepth);
	ifile.close();
	cerr << endl;

	VolSize = iWidth*iHeight*iDepth;
	image = new unsigned char [VolSize*nVolumes];
	unsigned char *ptr = image;

	ifile.open(path, ios::in);
	ifile.getline(vfile, MaxLine, '\n');
	while (ifile.getline(vfile, MaxLine, '\n') != NULL) {
		ReadVolume(dir, vfile, ptr);
		ptr += VolSize;
	}
	ifile.close();
}


/*************************[MAN-BEG]*******************************************
*
* NAME:
*   SetDisplayList
*
* DESCRIPTION:
*
*   Creates a display list containing all the enclosing rectangles 
*   corresponding to the cutplanes.
*
* FORMAL PARAMETERS:
*   None
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void SetDisplayList(void)
{
   float dist;
   int i;
   int nHSlices     = (sqrt(float(iWidth*iWidth + iHeight*iHeight + iDepth*iDepth)))/4.0;
   int nSlices      = 2*nHSlices + 1;
   int nElements    = nSlices*4;
   GLfloat *vPoints = new GLfloat [3*nElements];
   GLfloat *ptr     = vPoints;
   float dDist      = 1/128.0;
   float s          = 0.5;
   
   for (i = -nHSlices; i <= nHSlices; i++) {
	   dist     = i*dDist;

	   *(ptr++) = -s;
	   *(ptr++) = -s;
	   *(ptr++) =  dist;

	   *(ptr++) =  s;
	   *(ptr++) = -s;
	   *(ptr++) =  dist;

	   *(ptr++) =  s;
	   *(ptr++) =  s;
	   *(ptr++) =  dist;

	   *(ptr++) = -s;
	   *(ptr++) =  s;
	   *(ptr++) =  dist;
   }

   glEnableClientState(GL_VERTEX_ARRAY);
   glVertexPointer(3, GL_FLOAT, 0, vPoints);
   GL_DisplayList = glGenLists(1);
   glNewList(GL_DisplayList, GL_COMPILE);
   glDrawArrays(GL_QUADS, 0, nElements); 
   glEndList();
   delete [] vPoints;
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   InitColorMap
*
* DESCRIPTION:
*
*   Initializes color map
*
* FORMAL PARAMETERS:
*   None
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void InitColorMap(void)
{
	GLubyte val, *ptr = ColorMapArray = new GLubyte [ColorMapLength*4];
	int i;

	for (i = 0; i < ColorMapLength; i++) {
		*(ptr++) = RedTransferFunction[i];
		*(ptr++) = GreenTransferFunction[i];
		*(ptr++) = BlueTransferFunction[i];
		*(ptr++) = OpacityFunction[i];
	}
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   Init
*
* DESCRIPTION:
*
*   Initializes OpenGL state machine
*
* FORMAL PARAMETERS:
*   None
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void Init(void)
{    
   ReadVolumes("../Data/US/", "Baby.lst");
   InitColorMap();

   glClearColor (0.0, 0.0, 0.0, 0.0);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

   // Generate the 1D ColorMap
   glGenTextures(1, &GL_ColorMap);
   glBindTexture(GL_TEXTURE_1D, GL_ColorMap);
   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, ColorMapLength, 0, GL_RGBA, GL_UNSIGNED_BYTE, ColorMapArray);

   // Generate the ultrasound texture
   glGenTextures(1, &GL_USTexture);
   glBindTexture(GL_TEXTURE_3D, GL_USTexture);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexImage3DEXT(GL_TEXTURE_3D, 0, GL_RGBA, iWidth, iHeight, iDepth, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);

   glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
   glEnable(GL_BLEND);
   
   // Enable clip planes
   glEnable(GL_CLIP_PLANE0);
   glEnable(GL_CLIP_PLANE1);

   glEnable(GL_CLIP_PLANE2);
   glEnable(GL_CLIP_PLANE3);

   glEnable(GL_CLIP_PLANE4);
   glEnable(GL_CLIP_PLANE5);

   SetDisplayList();
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   Display
*
* DESCRIPTION:
*
*   Display routine for OpenGL
*
* FORMAL PARAMETERS:
*   None
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void Display(void)
{
   static unsigned char *ptr = image;
   static int index = 0;

   glEnable(GL_TEXTURE_3D);
   glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
   glBindTexture(GL_TEXTURE_3D, GL_USTexture);
   
   
   glTexSubImage3DEXT(GL_TEXTURE_3D, 0, 0, 0, 0, iWidth, iHeight, iDepth, GL_LUMINANCE, GL_UNSIGNED_BYTE, ptr);


   glClear(GL_COLOR_BUFFER_BIT);

   glPushMatrix ();
   glScalef(scale, scale, scale);

   glPushMatrix ();

   // Transform the viewing direction
   glRotatef(-gamma, 0.0, 0.0, 1.0);
   glRotatef(-theta, 0.0, 1.0, 0.0);
   glRotatef(-alpha, 1.0, 0.0, 0.0);
   glTranslatef(-0.5, -0.5, -0.5);

   // Send the values of the uniform variables to the vertex program
   float tanHApexAngle = tanl(d2r(ApexAngle_2));
   float Aperture_2 = Aperture/2.0;
   cgGLSetParameter4f(ZoomFactor, scale, scale, scale, 1.0);
   cgGLSetParameter4f(Pyramid, Aperture, 2.0*tanHApexAngle, 0.0, 0.0);
   cgGLSetParameter4f(ClipPlane0,  1.0,  0.0, tanHApexAngle, Aperture_2);
   cgGLSetParameter4f(ClipPlane1, -1.0,  0.0, tanHApexAngle, Aperture_2);
   cgGLSetParameter4f(ClipPlane2,  0.0,  1.0,  0.0, 0.5);
   cgGLSetParameter4f(ClipPlane3,  0.0, -1.0,  0.0, 0.5);
   cgGLSetParameter4f(ClipPlane4,  0.0,  0.0,  1.0, 0.0);
   cgGLSetParameter4f(ClipPlane5,  0.0,  0.0, -1.0, 1.0);

   // Send the model view matrix to the vertex program
   cgGLSetStateMatrixParameter(ModelView, CG_GL_MODELVIEW_MATRIX, CG_GL_MATRIX_INVERSE);

   glPopMatrix ();

   // Send the model view projection matrix to the vertex program
   cgGLSetStateMatrixParameter(ModelViewProj, CG_GL_MODELVIEW_PROJECTION_MATRIX, CG_GL_MATRIX_IDENTITY);
   
   // Render enclosing polygons
   glCallList(GL_DisplayList);
 
   glPopMatrix ();
   glutSwapBuffers();

   glDisable(GL_TEXTURE_3D);

   index++;
   ptr += VolSize;
   if (index == nVolumes) {
	   index = 0;
	   ptr = image;
   }
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   Idle
*
* DESCRIPTION:
*
*   Idle routine for OpenGL
*
* FORMAL PARAMETERS:
*   None
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void Idle(void)
{
	static clock_t thisClock, startClock = clock();
	static int numTimes = 0;
	int avgCount        = 20;
	float duration;

	if (numTimes == avgCount) {
		thisClock = clock();
		duration = (double)(thisClock - startClock)/ CLOCKS_PER_SEC;
		startClock = thisClock;
		numTimes = 0;
		cerr << "Frame Rate = " << ((float)avgCount)/duration << endl;
	}

	Display();

	if (!step_p) {
		theta += 2.0;
		theta = (theta >= 360.0) ? 0.0 : theta;
	}

	numTimes++;
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   Reshape
*
* DESCRIPTION:
*
*   Reshape routine for OpenGL
*
* FORMAL PARAMETERS:
*   w and h are width and height of the window
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void Reshape(int w, int h)
{
   glViewport(0, 0, (GLsizei) w, (GLsizei) h);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();

   GLfloat size_ = 1.0;
   GLfloat aspect = (GLfloat) h/(GLfloat) w;
   glOrtho(-size_, size_, -size_*aspect, size_*aspect, -size_, size_);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   PrintHelp
*
* DESCRIPTION:
*
*   Prints a help line
*
* FORMAL PARAMETERS:
*   None
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void PrintHelp(void)
{
	cerr << endl;
	cerr << "Help:" << endl;
	cerr << endl;
	cerr << "x/X: Increase or decrease Alpha" << endl;
	cerr << "y/Y: Increase or decrease Theta" << endl;
	cerr << "g/G: Increase or decrease Gamma" << endl;
	cerr << "z/Z: Zoom in or out" << endl;
	cerr << "a/A: Decrease/increase aperture size" << endl;
	cerr << "b/B: Decrease/increase half of apex angle" << endl;
	cerr << "s  : Stop/start automatic rotation" << endl;
	cerr << "ESC: Quit" << endl;
	cerr << endl;
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   CheckCgError
*
* DESCRIPTION:
*
*   This reports the Cg error messages
*
* FORMAL PARAMETERS:
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
static void CheckCgError(void)
{
	CGerror err = cgGetError();
	
	if (err != CG_NO_ERROR) {
		printf("CG error: %s\n", cgGetErrorString(err));
		exit(1);
	}
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   InitializeFragmentProgram
*
* DESCRIPTION:
*
*   This initializes the fragment program
*
* FORMAL PARAMETERS:
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void InitializeVertexProgram(void)
{
	vProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
	cgGLSetOptimalOptions(vProfile);
	
	vProgram = cgCreateProgramFromFile(Context, CG_SOURCE, "VolumeRenderPyramidUSV.cg", vProfile, "VertexProgram", 0);
	CheckCgError();
	
	cerr << "---- VERTEX PROGRAM BEGIN ----" << endl;
	cerr << cgGetProgramString(vProgram, CG_COMPILED_PROGRAM) << endl;
	cerr << "---- VERTEX PROGRAM END ----" << endl;
	
	if(vProgram != NULL) { 
		cgGLLoadProgram(vProgram);
		CheckCgError();
		ZoomFactor    = cgGetNamedParameter(vProgram, "ZoomFactor");
		CheckCgError();
		Pyramid       = cgGetNamedParameter(vProgram, "Pyramid");
		CheckCgError();
		ClipPlane0    = cgGetNamedParameter(vProgram, "ClipPlane0");
		CheckCgError();
		ClipPlane1    = cgGetNamedParameter(vProgram, "ClipPlane1");
		CheckCgError();
		ClipPlane2    = cgGetNamedParameter(vProgram, "ClipPlane2");
		CheckCgError();
		ClipPlane3    = cgGetNamedParameter(vProgram, "ClipPlane3");
		CheckCgError();
		ClipPlane4    = cgGetNamedParameter(vProgram, "ClipPlane4");
		CheckCgError();
		ClipPlane5    = cgGetNamedParameter(vProgram, "ClipPlane5");
		CheckCgError();
		ModelView     = cgGetNamedParameter(vProgram, "ModelView");
		CheckCgError();
		ModelViewProj = cgGetNamedParameter(vProgram, "ModelViewProj");
		CheckCgError();
	}
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   InitializeFragmentProgram
*
* DESCRIPTION:
*
*   This initializes the fragment program
*
* FORMAL PARAMETERS:
*   if Source_p is true, we will load the source program. Else load the
*   object program.
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void InitializeFragmentProgram(char Source_p)
{
	fProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
	cgGLSetOptimalOptions(fProfile);

	if (Source_p)
		fProgram = cgCreateProgramFromFile(Context, CG_SOURCE, "VolumeRenderPyramidUSF.cg", fProfile, "FragmentProgram", 0);
	else
		fProgram = cgCreateProgramFromFile(Context, CG_OBJECT, "VolumeRenderPyramidUSF.ocg", fProfile, "FragmentProgram", 0);
	CheckCgError();

	cerr << "---- FRAGMENT PROGRAM BEGIN ----" << endl;
	cerr << cgGetProgramString(fProgram, CG_COMPILED_PROGRAM) << endl;
	cerr << "---- FRAGMENT PROGRAM END ----" << endl;

	if(fProgram != NULL) {
		cgGLLoadProgram(fProgram);
		CheckCgError();
		USTexture = cgGetNamedParameter(fProgram, "USTexture");
		CheckCgError();
		ColorMap = cgGetNamedParameter(fProgram, "ColorMap");
		CheckCgError();
	}

	cgGLSetTextureParameter(ColorMap, GL_ColorMap);
	CheckCgError();
	cgGLSetTextureParameter(USTexture, GL_USTexture);
	CheckCgError();
}


/*************************[MAN-BEG]*******************************************
*
* NAME:
*   InitializeGPU
*
* DESCRIPTION:
*
*   This initializes the GPU
*
* FORMAL PARAMETERS:
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/	
void InitializeGPU(void)
{
   	// Create Cg Context
   	Context = cgCreateContext();
	CheckCgError();

	InitializeVertexProgram();
	cgGLEnableProfile(vProfile);
	CheckCgError();
	cgGLBindProgram(vProgram);
	CheckCgError();


	char Source_p = 1;
	// To load the Cg source code, comment out the following line.
	Source_p = 0;
	InitializeFragmentProgram(Source_p);
	cgGLEnableProfile(fProfile);
	CheckCgError();
	cgGLBindProgram(fProgram);
	CheckCgError();
	cgGLEnableTextureParameter(ColorMap);
	CheckCgError();
	cgGLEnableTextureParameter(USTexture);
	CheckCgError();
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   QuitGPU
*
* DESCRIPTION:
*
*   This quits the GPU
*
* FORMAL PARAMETERS:
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void QuitGPU(void)
{
	cgDestroyProgram(vProgram);
	cgGLDisableTextureParameter(ColorMap);
	CheckCgError();	
	cgGLDisableTextureParameter(USTexture);
	CheckCgError();
	cgGLDisableProfile(fProfile);
	CheckCgError();
	cgDestroyContext(Context);
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   Keyboard
*
* DESCRIPTION:
*
*   Handles keybord commands
*
* FORMAL PARAMETERS:
*   key is the key pressed by the user.
*   (x, y) are the coordinates of the mouse
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
void Keyboard(unsigned char key, int x, int y)
{
	switch (key) {
	case 'h':
		PrintHelp();
		break;
	case 'x':
		alpha += 10.0;
		alpha = (alpha >= 360.0) ? 0.0 : alpha;
		cerr << "Alpha: " << alpha << endl;
		break;
	case 'X':
		alpha -= 10.0;
		alpha = (alpha < 0.0) ? 360.0 : alpha;
		cerr << "Alpha: " << alpha << endl;
		break;
	case 'y':
		theta += 10.0;
		theta = (theta >= 360.0) ? 0.0 : theta;
		cerr << "Theta: " << theta << endl;
		break;
	case 'Y':
		theta -= 10.0;
		theta = (theta < 0.0) ? 360.0 : theta;
		cerr << "Theta: " << theta << endl;
		break;
	case 'g':
		gamma += 10.0;
		gamma = (gamma >= 360.0) ? 0.0 : gamma;
		cerr << "Gamma: " << gamma << endl;
		break;
	case 'G':
		gamma -= 10.0;
		gamma = (gamma < 0.0) ? 360.0 : gamma;
		cerr << "Gamma: " << gamma << endl;
		break;
	case 'z':
		scale *= 0.9;
		cerr << "ZoomFactor: " << scale << endl;
		break;
	case 'Z':
		scale *= 1.1;
		cerr << "ZoomFactor: " << scale << endl;
		break;
	case 's':
		step_p = !step_p;
		break;
	case 'a':
		Aperture -= 0.01;
		cerr << "Aperture: " << Aperture << endl;
		break;
	case 'A':
		Aperture += 0.01;
		cerr << "Aperture: " << Aperture << endl;
		break;
	case 'b':
		ApexAngle_2 -= 2.0;
		cerr << "Half of apex Angle: " << ApexAngle_2 << endl;
		break;
	case 'B':
		ApexAngle_2 += 2.0;
		cerr << "Half of apex Angle: " << ApexAngle_2 << endl;
		break;
	case 27:
		delete [] image;
		QuitGPU();
	    exit(0);
		break;
	}
}

/*************************[MAN-BEG]*******************************************
*
* NAME:
*   main
*
* DESCRIPTION:
*
* This is the main program that starts the app.
*
* FORMAL PARAMETERS:
*  none
*
* RETURNS:
*   None
*
* REVISION HISTORY:
* Rev     When      Who         What
* A	   16May03      sumane		Created.
**************************[MAN-END]******************************************/
int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(640, 480);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(argv[0]);
	
	if (glutExtensionSupported("GL_EXT_texture3D")) {
		cerr << "GL_EXT_texture3D supported!" << endl;
		glTexImage3DEXT    = (PFNGLTEXIMAGE3DEXTPROC)wglGetProcAddress("glTexImage3DEXT");
		glTexSubImage3DEXT = (PFNGLTEXSUBIMAGE3DEXTPROC)wglGetProcAddress("glTexSubImage3DEXT");
	}
	else {
		cerr << "GL_EXT_texture3D unsupported!" << endl;
		exit(0);
	}
	
	Init();
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutIdleFunc(Idle);
	glutKeyboardFunc (Keyboard);
	
	InitializeGPU();
	
	glutMainLoop();
	
	QuitGPU();

	return 0; 
}

