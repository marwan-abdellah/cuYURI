#include "Glut.h"

#include <glut.h>
void OpenGL::InitGlut(int argc, char** argv)
{
    INFO("Initializing GLUT");

    int initialWinWidth = 256;
    int initialWinHeight = 256;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(initialWinWidth, initialWinHeight);
    glutCreateWindow("cuYURI - OpenGL Integration");

    INFO("Display Mode : GLUT_RGBA | GLUT_DOUBLE");
    INFO("Initializing GLUT DONE");
}
