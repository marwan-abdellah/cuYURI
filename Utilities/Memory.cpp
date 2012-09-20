/*
 * Memory.cpp
 *
 *  Created on: May 23, 2012
 *      Author: abdellah
 */

#include "Memory.h"

char* Memory::alloc_1D_char(const int NX)
{
	LOG();

    char* data = (char*) malloc (sizeof(char) * NX);
	return data;
}

float* Memory::alloc_1D_float(const int NX)
{
	LOG();
    float* data = (float*) malloc (sizeof(float) * NX);
	return data;
}
double* Memory::alloc_1D_double(const int NX)
{
	LOG();
    double* data = (double*) malloc (sizeof(double) * NX);
	return data;
}

char** Memory::alloc_2D_char(const int NX, const int NY)
{
	LOG();

	char** data;
    data = (char**) malloc (sizeof(char*) * NX);
    for(int i = 0; i < NX; i++)
        data[i] = (char*) malloc (sizeof(char) * NY);

	return data;
}

float** Memory::alloc_2D_float(const int NX, const int NY)
{
	LOG();

	float** data;
    data = (float**) malloc (sizeof(float*) * NX);
    for(int i = 0; i < NX; i++)
        data[i] = (float*) malloc (sizeof(float) * NY);

	return data;
}
double** Memory::alloc_2D_double(const int NX, const int NY)
{
	LOG();

	double** data;
    data = (double**) malloc (sizeof(double) * NX);
    for(int i = 0; i < NX; i++)
        data[i] = (double*) malloc (sizeof(double) * NY);

	return data;
}

char*** Memory::alloc_3D_char(const int NX, const int NY, const int NZ)
{
	LOG();

	char*** data;
    data = (char***) malloc (sizeof(char**) * NX);
    for(int i = 0; i < NX; i++)
	{
        data[i] = (char**) malloc (sizeof(char*) * NY);
        for (int j = 0; j < NY; j++)
            data[i][j] = (char*) malloc (sizeof(char) * NZ);
	}

	return data;
}

float*** Memory::alloc_3D_float(const int NX, const int NY, const int NZ)
{
	LOG();

	float*** data;
    data = (float***) malloc (sizeof(float**) * NX);
    for(int i = 0; i < NX; i++)
	{
        data[i] = (float**) malloc (sizeof(float*) * NY);
        for (int j = 0; j < NY; j++)
            data[i][j] = (float*) malloc (sizeof(float) * NZ);
	}

	return data;
}

double*** Memory::alloc_3D_double(const int NX, const int NY, const int NZ)
{
	LOG();

	double*** data;
    data = (double***) malloc (sizeof(double**) * NX);
    for(int i = 0; i < NX; i++)
	{
        data[i] = (double**) malloc (sizeof(double*) * NY);
        for (int j = 0; j < NY; j++)
            data[i][j] = (double*) malloc (sizeof(double) * NZ);
	}

	return data;
}

cufftComplex** Memory::alloc_2D_cufftComplex(const int NX, const int NY)
{
	LOG();

	cufftComplex** data;
    data = (cufftComplex**) malloc (sizeof(cufftComplex) * NX);
    for(int i = 0; i < NX; i++)
        data[i] = (cufftComplex*) malloc (sizeof(cufftComplex) * NY);

	return data;
}

cufftDoubleComplex** Memory::alloc_2D_cufftDoubleComplex(const int NX, const int NY)
{
	LOG();

	cufftDoubleComplex** data;
    data = (cufftDoubleComplex**) malloc (sizeof(cufftDoubleComplex) * NX);
    for(int i = 0; i < NX; i++)
        data[i] = (cufftDoubleComplex*) malloc (sizeof(cufftDoubleComplex) * NY);

	return data;
}

cufftComplex*** Memory::alloc_3D_cufftComplex(const int NX, const int NY, const int NZ)
{
	LOG();

	cufftComplex*** data;
    data = (cufftComplex***) malloc (sizeof(cufftComplex**) * NX);
    for(int i = 0; i < NX; i++)
	{
        data[i] = (cufftComplex**) malloc (sizeof(cufftComplex*) * NY);
        for (int j = 0; j < NY; j++)
            data[i][j] = (cufftComplex*) malloc (sizeof(cufftComplex) * NZ);
	}

	return data;
}

cufftDoubleComplex*** Memory::alloc_3D_cufftDoubleComplex(const int NX, const int NY, const int NZ)
{
	LOG();

	cufftDoubleComplex*** data;
    data = (cufftDoubleComplex***) malloc (sizeof(cufftDoubleComplex**) * NX);
    for(int i = 0; i < NX; i++)
	{
        data[i] = (cufftDoubleComplex**) malloc (sizeof(cufftDoubleComplex*) * NY);
        for (int j = 0; j < NY; j++)
            data[i][j] = (cufftDoubleComplex*) malloc (sizeof(cufftDoubleComplex) * NZ);
	}

	return data;
}

fftwf_complex** Memory::alloc_2D_fftwfComplex(const int NX, const int NY)
{
	LOG();

	fftwf_complex** data;
    data = (fftwf_complex**) malloc (sizeof(fftwf_complex) * NX);
    for(int i = 0; i < NX; i++)
        data[i] = (fftwf_complex*) malloc (sizeof(fftwf_complex) * NY);

	return data;
}

fftw_complex** Memory::alloc_2D_fftwComplex(const int NX, const int NY)
{
	LOG();

	fftw_complex** data;
    data = (fftw_complex**) malloc (sizeof(fftw_complex) * NX);
    for(int i = 0; i < NX; i++)
        data[i] = (fftw_complex*) malloc (sizeof(fftw_complex) * NY);

	return data;
}


fftwf_complex*** Memory::alloc_3D_fftwfComplex(const int NX, const int NY, const int NZ)
{
	LOG();

	fftwf_complex*** data;
    data = (fftwf_complex***) malloc (sizeof(fftwf_complex**) * NX);
    for(int i = 0; i < NX; i++)
	{
        data[i] = (fftwf_complex**) malloc (sizeof(fftwf_complex*) * NY);
        for (int j = 0; j < NY; j++)
            data[i][j] = (fftwf_complex*) malloc (sizeof(fftwf_complex) * NZ);
	}

	return data;
}

fftw_complex*** Memory::alloc_3D_fftwComplex(const int NX, const int NY, const int NZ)
{
	LOG();

	fftw_complex*** data;
    data = (fftw_complex***) malloc (sizeof(fftw_complex**) * NX);
    for(int i = 0; i < NX; i++)
	{
        data[i] = (fftw_complex**) malloc (sizeof(fftw_complex*) * NY);
        for (int j = 0; j < NY; j++)
            data[i][j] = (fftw_complex*) malloc (sizeof(fftw_complex) * NZ);
	}

	return data;
}

void Memory::free_2D_float(float** ptrData, const int NX, const int NY)
{
    for (int i = 0; i < NY; i++)
		free(ptrData[i]);

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_2D_double(double** ptrData, const int NX, const int NY)
{
    for (int i = 0; i < NY; i++)
		free(ptrData[i]);

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_3D_float(float*** ptrData, const int NX, const int NY, const int NZ)
{
    for(int i = 0; i < NY; i++)
    {
        for(int j = 0; j < NZ; j++)
                free((void*) ptrData[i][j]);
        free((void*)ptrData[i]);
    }

    free((void*) ptrData);
    ptrData = NULL;
}

void Memory::free_3D_char(char*** ptrData, const int NX, const int NY, const int NZ)
{
    for(int i = 0; i < NY; i++)
    {
        for(int j = 0; j < NZ; j++)
                free((void*) ptrData[i][j]);
        free((void*)ptrData[i]);
    }

    free((void*) ptrData);
    ptrData = NULL;
}

void Memory::free_3D_double(double*** ptrData, const int NX, const int NY, const int NZ)
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

void Memory::free_2D_cufftComplex(cufftComplex** ptrData, const int NX, const int NY)
{
    for (int i = 0; i < NY; i++)
		free(ptrData[i]);

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_2D_cufftDoubleComplex(cufftDoubleComplex** ptrData, const int NX, const int NY)
{
    for (int i = 0; i < NY; i++)
		free(ptrData[i]);

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_3D_cufftComplex(cufftComplex*** ptrData, const int NX, const int NY, int NZ)
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

void Memory::free_3D_cufftDoubleComplex(cufftDoubleComplex*** ptrData, const int NX, const int NY, int NZ)
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

void Memory::free_2D_fftwfComplex(fftwf_complex** ptrData, const int NX, const int NY)
{
    for (int i = 0; i < NY; i++)
		free(ptrData[i]);

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_2D_fftwComplex(fftw_complex** ptrData, const int NX, const int NY)
{
    for (int i = 0; i < NY; i++)
		free(ptrData[i]);

	free(ptrData);
	ptrData = NULL;
}

void Memory::free_3D_fftwfComplex(fftwf_complex*** ptrData, const int NX, const int NY, int NZ)
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

void Memory::free_3D_fftwComplex(fftw_complex*** ptrData, const int NX, const int NY, int NZ)
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


