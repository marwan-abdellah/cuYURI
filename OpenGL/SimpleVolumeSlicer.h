#ifndef _SIMPLE_VOLUME_SLICER_H_
#define _SIMPLE_VOLUME_SLICER_H_


namespace SimpleVolumeSlicer
{
void ReadHeader(char *prefix, int &w, int &h, int &d);
void SetDisplayList(void);
void Init(void);
void Display(void);
void Idle(void);
void Reshape(int w, int h);
void PrintHelp(void);
void Keyboard(unsigned char key, int x, int y);
void runIt(int argc, char** argv);

}


#endif // _SIMPLE_VOLUME_SLICER_H_
