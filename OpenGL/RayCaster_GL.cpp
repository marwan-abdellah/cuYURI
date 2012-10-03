#include "OpenGL.h"

void OpenGL::RayCaster::Resize_GL(int windowWidth, int windowHeight)
{
    // For initialization checking
    if (windowHeight == 0)
        windowHeight = 1;

    // Adjust the viewport
    glViewport(0, 0, windowWidth, windowHeight);

    // Get the projection matrix
    glMatrixMode(GL_PROJECTION);

    // Load I
    glLoadIdentity();

    // Adjust the perspective to the new widows parametsrs
    gluPerspective(60.0, (GLfloat)windowWidth / (GLfloat)windowHeight, 0.01, 400.0);

    // Call the model view matrix on top of the stack
    glMatrixMode(GL_MODELVIEW);
}



// for contiunes keypresses
void ProcessKeys()
{
    // Process keys
    for (int i = 0; i < 256; i++)
    {
        if (!gKeys[i])  { continue; }
        switch (i)
        {
        case ' ':
            break;
        case 'w':
            stepsize += 1.0/2048.0;
            if(stepsize > 0.25) stepsize = 0.25;
            break;
        case 'e':
            stepsize -= 1.0/2048.0;
            if(stepsize <= 1.0/200.0) stepsize = 1.0/200.0;
            break;
        }
    }

}

void key(unsigned char k, int x, int y)
{
    gKeys[k] = true;
}

void KeyboardUpCallback(unsigned char key, int x, int y)
{
    gKeys[key] = false;

    switch (key)
    {
    case 27 :
        {
            exit(0); break;
        }
    case ' ':
        toggle_visuals = !toggle_visuals;
        break;
    }
}

void OpenGL::RayCaster::Idel_GL()
{
    ProcessKeys();
    glutPostRedisplay();
}
