#include "Memory_3D.h"

template <typename T>
extern
T*** Memory::Alloc_3D_Array(const int NX, const int NY, const int NZ)
{
    T*** data;
    data = (T***) malloc (sizeof(T**) * NX);
    for(int i = 0; i < NX; i++)
    {
        data[i] = (T**) malloc (sizeof(T*) * NY);
        for (int j = 0; j < NY; j++)
            data[i][j] = (T*) malloc (sizeof(T) * NZ);
    }

    return data;
}

template <typename T>
extern
void Memory::Free_3D_Array(T*** ptrData, const int NX, const int NY, int NZ)
{
    for(int i = 0; i < NY; i++)
    {
        for(int j = 0; j < NZ; j++)
                free(ptrData[i][j]);
        free(ptrData[i]);
    }

    free(ptrData);
    ptrData = NULL;
}

template
char*** Memory::Alloc_3D_Array <char>
(const int , const int NY, const int NZ);

template
unsigned char*** Memory::Alloc_3D_Array <unsigned char>
(const int , const int NY, const int NZ);

template
int*** Memory::Alloc_3D_Array <int>
(const int NX, const int NY, const int NZ);

template
unsigned int*** Memory::Alloc_3D_Array <unsigned int>
(const int NX, const int NY, const int NZ);

template
float*** Memory::Alloc_3D_Array <float>
(const int NX, const int NY, const int NZ);

template
double*** Memory::Alloc_3D_Array <double>
(const int NX, const int NY, const int NZ);

template
fftwf_complex*** Memory::Alloc_3D_Array <fftwf_complex>
(const int NX, const int NY, const int NZ);

template
fftw_complex*** Memory::Alloc_3D_Array <fftw_complex>
(const int NX, const int NY, const int NZ);

template
cufftComplex*** Memory::Alloc_3D_Array <cufftComplex>
(const int NX, const int NY, const int NZ);

template
cufftDoubleComplex*** Memory::Alloc_3D_Array <cufftDoubleComplex>
(const int NX, const int NY, const int NZ);

template
void Memory::Free_3D_Array <char>
(char*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <unsigned char>
(unsigned char*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <int>
(int*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <unsigned int>
(unsigned int*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <float>
(float*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <double>
(double*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <fftwf_complex>
(fftwf_complex*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <fftw_complex>
(fftw_complex*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <cufftComplex>
(cufftComplex*** ptrData, const int NX, const int NY, int NZ);

template
void Memory::Free_3D_Array <cufftDoubleComplex>
(cufftDoubleComplex*** ptrData, const int NX, const int NY, int NZ);
