#ifndef _VOLUME_SLICER_H_
#define _VOLUME_SLICER_H_


namespace VolumeSlicer
{
void ReadHeader(char *prefix, int &w, int &h, int &d);
void SetDisplayList(void);
void Init(char** argv);
void Display(void);
void Idle(void);
void Reshape(int w, int h);
void PrintHelp(void);
void Keyboard(unsigned char key, int x, int y);
void InitGlut(int argc, char** argv);
void RegisterOpenGLCallBacks();
void UpdateScene();
void runRenderingEngine(int argc, char** argv);
}


#endif // _VOLUME_SLICER_H_
