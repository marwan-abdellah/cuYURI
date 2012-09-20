#include "Memory_1D.h"

template <typename T>
extern
T* Memory::Alloc_1D_Array(const int NX)
{
    T* data = (T*) malloc (sizeof(T) * NX);
    return data;
}

template <typename T>
extern
void Memory::Free_1D_Array(T* ptrData)
{
    free(ptrData);
    ptrData = NULL;
}

template
char* Memory::Alloc_1D_Array <char>
(const int NX);

template
unsigned char* Memory::Alloc_1D_Array <unsigned char>
(const int NX);

template
int* Memory::Alloc_1D_Array <int>
(const int NX);

template
unsigned int* Memory::Alloc_1D_Array <unsigned int>
(const int NX);

template
float* Memory::Alloc_1D_Array <float>
(const int NX);

template
double* Memory::Alloc_1D_Array <double>
(const int NX);

template fftwf_complex*
Memory::Alloc_1D_Array <fftwf_complex>
(const int NX);

template
fftw_complex* Memory::Alloc_1D_Array <fftw_complex>
(const int NX);

template
cufftComplex* Memory::Alloc_1D_Array <cufftComplex>
(const int NX);

template
cufftDoubleComplex* Memory::Alloc_1D_Array <cufftDoubleComplex>
(const int NX);

template
void Memory::Free_1D_Array <char>
(char* ptrData);

template
void Memory::Free_1D_Array <unsigned char>
(unsigned char* ptrData);

template
void Memory::Free_1D_Array <int>
(int* ptrData);

template
void Memory::Free_1D_Array <unsigned int>
(unsigned int* ptrData);

template
void Memory::Free_1D_Array <float>
(float* ptrData);

template
void Memory::Free_1D_Array <double>
(double* ptrData);

template
void Memory::Free_1D_Array <fftwf_complex>
(fftwf_complex* ptrData);

template
void Memory::Free_1D_Array <fftw_complex>
(fftw_complex* ptrData);

template
void Memory::Free_1D_Array <cufftComplex>
(cufftComplex* ptrData);

template
void Memory::Free_1D_Array <cufftDoubleComplex>
(cufftDoubleComplex* ptrData);
