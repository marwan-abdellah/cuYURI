/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: arr_2D_FLAT.cpp
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#include "arr_2D_FLAT.h"

#define TYPE float

namespace arr_2D
{
	namespace flat
	{
		template <typename T>
		T* createArray(int arrayWidth, int arrayHeight);

		template int* createArray(int arrayWidth, int arrayHeight); 
		template float* createArray(int arrayWidth, int arrayHeight);
		template double* createArray(int arrayWidth, int arrayHeight);

		template <typename T>
		void fillArray_SEQ(T* in, int arrayWidth, int arrayHeight);

		template void fillArray_SEQ(int* in, int arrayWidth, int arrayHeight);
		template void fillArray_SEQ(float* in, int arrayWidth, int arrayHeight);
		template void fillArray_SEQ(double* in, int arrayWidth, int arrayHeight);

		template <typename T>
		void fillArray_RND(T* in, int arrayWidth, int arrayHeight);

		template void fillArray_RND(int* in, int arrayWidth, int arrayHeight);
		template void fillArray_RND(float* in, int arrayWidth, int arrayHeight);
		template void fillArray_RND(double* in, int arrayWidth, int arrayHeight);

		template <typename T>
		void zeroArray(T* in, int arrayWidth, int arrayHeight);

		template void zeroArray(int* in, int arrayWidth, int arrayHeight);
		template void zeroArray(float* in, int arrayWidth, int arrayHeight);
		template void zeroArray(double* in, int arrayWidth, int arrayHeight);

		template <typename T>
		void printArray(T* in, int arrayWidth, int arrayHeight, bool trace);

		template void printArray(int* in, int arrayWidth, int arrayHeight, bool trace);
		template void printArray(float* in, int arrayWidth, int arrayHeight, bool trace);
		template void printArray(double* in, int arrayWidth, int arrayHeight, bool trace);

		template <typename T>
		T* flatArray(T** in, int arrayWidth, int arrayHeight);

		template int* flatArray(int** in, int arrayWidth, int arrayHeight);
		template float* flatArray(float** in, int arrayWidth, int arrayHeight);
		template double* flatArray(double** in, int arrayWidth, int arrayHeight);

		template <typename T>
		void clearArray(T* in);

		template void clearArray(int* in);
		template void clearArray(float* in);
		template void clearArray(double* in);

		/* @addArrays */
		template <typename T> 
		void addArrays(T* in_1, T* in_2, T* out, int arrayWidth, int arrayHeight); 

		/* Template specialization */
		template void addArrays(int* in_1, int* in_2, int* out, int arrayWidth, int arrayHeight); 
		template void addArrays(float* in_1, float* in_2, float* out, int arrayWidth, int arrayHeight);
		template void addArrays(double* in_1, double* in_2, double* out, int arrayWidth, int arrayHeight);

		/* @subArrays */
		template <typename T> 
		void subArrays(T* in_1, T* in_2, T* out, int arrayWidth, int arrayHeight); 

		/* Template specialization */
		template void subArrays(int* in_1, int* in_2, int* out, int arrayWidth, int arrayHeight); 
		template void subArrays(float* in_1, float* in_2, float* out, int arrayWidth, int arrayHeight);
		template void subArrays(double* in_1, double* in_2, double* out, int arrayWidth, int arrayHeight);

		/* @mulArrays */
		template <typename T> 
		void mulArrays(T* in_1, T* in_2, T* out, int arrayWidth, int arrayHeight); 

		/* Template specialization */
		template void mulArrays(int* in_1, int* in_2, int* out, int arrayWidth, int arrayHeight); 
		template void mulArrays(float* in_1, float* in_2, float* out, int arrayWidth, int arrayHeight);
		template void mulArrays(double* in_1, double* in_2, double* out, int arrayWidth, int arrayHeight);

		/* @divArrays */
		template <typename T> 
		void divArrays(T* in_1, T* in_2, T* out, int arrayWidth, int arrayHeight); 

		/* Template specialization */
		template void divArrays(int* in_1, int* in_2, int* out, int arrayWidth, int arrayHeight); 
		template void divArrays(float* in_1, float* in_2, float* out, int arrayWidth, int arrayHeight);
		template void divArrays(double* in_1, double* in_2, double* out, int arrayWidth, int arrayHeight);
	}
}

int arr_2D::flat::get2DIndex(int iElement, int jElement, int arrayWidth)
{
	return (iElement + (jElement * arrayWidth));	
}

template <typename T> 
T* arr_2D::flat::createArray(int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	T* array_2D;
	array_2D = (T*) malloc(arrayWidth * arrayHeight * sizeof(T)); 

	return array_2D; 
}

template <typename T> 
void arr_2D::flat::fillArray_SEQ(T* in, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	for (int i = 0; i < (arrayWidth * arrayHeight); i++)
		in[i] = (T) (i); 
}

template <typename T> 
void arr_2D::flat::fillArray_RND(T* in, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	for (int i = 0; i < (arrayWidth * arrayHeight); i++)
		in[i] = (T) rand(); 
}

template <typename T> 
void arr_2D::flat::zeroArray(T* in, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	for (int i = 0; i < (arrayWidth * arrayHeight); i++)
		in[i] = (T) 0; 
}

template <typename T> 
void arr_2D::flat::printArray(T* in, int arrayWidth, int arrayHeight, bool trace)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	std::string resString; 
	std::string traceString = ""; 

	int index = 0; 

	if (trace)
	{
		for (int j = 0; j < arrayHeight; j++)
		{
			for (int i = 0; i < arrayWidth; i++)
			{
				index =arr_2D::flat::get2DIndex(i, j, arrayWidth); 

				/* @space for "escape sequence" */
				/*
				traceString =	util::charArrayToString("@[") + \
								util::intToString(i) + \
								util::charArrayToString("]") + \
								util::charArrayToString("[") + \
								util::intToString(j) + \
								util::charArrayToString("] = ") + \
								util::floatToString(in[index]);  
				*/
				
				traceString = util::floatToString(in[index]); 
				traceFile::trace(traceString);
			}
			/* @New line*/
			traceFile::trace(" \n");
		}
		
		for (int i = 0; i < arrayWidth; i++)
			traceFile::trace("___");

		/* @New line*/
		traceFile::trace(" \n\n");
	}
	else 
	{
		for (int j = 0; j < arrayHeight; j++)
			for (int i = 0; i < arrayWidth; i++)
			{
				index =arr_2D::flat::get2DIndex(i, j, arrayWidth); 

				printf("@[%d][%d]= %f \n", i, j, in[index]);
			}
	}
}

template <typename T>
T* arr_2D::flat::flatArray(T** in, int arrayWidth, int arrayHeight)
{
	T* out; 
	out = arr_2D::flat::createArray <T> (arrayWidth, arrayHeight);

	int index = 0; 

	for (int j = 0; j < arrayHeight; j++)
		for (int i = 0; i < arrayWidth; i++)	
			{
				index =arr_2D::flat::get2DIndex(i, j, arrayWidth); 

				out[index] = in[i][j]; 
			}

	return out; 
}	

template <typename T>
void arr_2D::flat::clearArray(T* in)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	if (in)
        free(in);
}

template <typename T> 
void arr_2D::flat::addArrays(T* in_1, T* in_2, T* out, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	int index = 0; 

	for (int j = 0; j < arrayHeight; j++)
		for (int i = 0; i < arrayWidth ; i++)
		{
			index =arr_2D::flat::get2DIndex(i, j, arrayWidth);

			out[index] = in_1[index] + in_2[index]; 
		}
}

template <typename T> 
void arr_2D::flat::subArrays(T* in_1, T* in_2, T* out, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	int index = 0; 

	for (int j = 0; j < arrayHeight; j++)
		for (int i = 0; i < arrayWidth ; i++)
		{
			index =arr_2D::flat::get2DIndex(i, j, arrayWidth);
			out[index] = in_1[index] - in_2[index]; 
		}
}

template <typename T> 
void arr_2D::flat::mulArrays(T* in_1, T* in_2, T* out, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	int index = 0; 

	for (int j = 0; j < arrayHeight; j++)
		for (int i = 0; i < arrayWidth ; i++)
		{
			index =arr_2D::flat::get2DIndex(i, j, arrayWidth);

			out[index] = in_1[index] * in_2[index]; 
		}
}

template <typename T> 
void arr_2D::flat::divArrays(T* in_1, T* in_2, T* out, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	int index = 0; 

	for (int j = 0; j < arrayHeight; j++)
		for (int i = 0; i < arrayWidth ; i++)
		{
			index = arr_2D::flat::get2DIndex(i, j, arrayWidth);

			out[index] = in_1[index] / in_2[index]; 
		}
}