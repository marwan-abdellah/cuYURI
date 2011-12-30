/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: arr_2D_SQUARE.cpp
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#include "arr_2D_SQUARE.h"

#define TYPE float
#define getVarName(varName,holder) sprintf(holder, "%s", #varName)

namespace arr_2D
{
	namespace square
	{
		int get2DIndex(int iElement, int jElement, int arrayWidth);

		template <typename T>
		T** createArray(int arrayWidth, int arrayHeight);
		
		template int** createArray(int arrayWidth, int arrayHeight);
		template float** createArray(int arrayWidth, int arrayHeight);
		template double** createArray(int arrayWidth, int arrayHeight);

		template <typename T>
		void fillArray_SEQ(T** in, int arrayWidth, int arrayHeight);

		template void fillArray_SEQ(int** in, int arrayWidth, int arrayHeight);
		template void fillArray_SEQ(float** in, int arrayWidth, int arrayHeight);
		template void fillArray_SEQ(double** in, int arrayWidth, int arrayHeight);

		template <typename T>
		void fillArray_RND(T** in, int arrayWidth, int arrayHeight);

		template void fillArray_RND(int** in, int arrayWidth, int arrayHeight);
		template void fillArray_RND(float** in, int arrayWidth, int arrayHeight);
		template void fillArray_RND(double** in, int arrayWidth, int arrayHeight);

		template <typename T>
		void zeroArray(T** in, int arrayWidth, int arrayHeight);
		
		template void zeroArray(int** in, int arrayWidth, int arrayHeight);
		template void zeroArray(float** in, int arrayWidth, int arrayHeight);
		template void zeroArray(double** in, int arrayWidth, int arrayHeight);

		template <typename T>
		void printArray(T** in, int arrayWidth, int arrayHeight, bool trace);
		
		template void printArray(int** in, int arrayWidth, int arrayHeight, bool trace);
		template void printArray(float** in, int arrayWidth, int arrayHeight, bool trace);
		template void printArray(double** in, int arrayWidth, int arrayHeight, bool trace);

		template <typename T>
		T** squareArray(T* in, int arrayWidth, int arrayHeight); 

		template int** squareArray(int* in, int arrayWidth, int arrayHeight); 
		template float** squareArray(float* in, int arrayWidth, int arrayHeight); 
		template double** squareArray(double* in, int arrayWidth, int arrayHeight); 

		template <typename T>
		void clearArray(T** in, int arrayWidth, int arrayHeight);

		template void clearArray(int** in, int arrayWidth, int arrayHeight);
		template void clearArray(float** in, int arrayWidth, int arrayHeight);
		template void clearArray(double** in, int arrayWidth, int arrayHeight);

		/* @addArrays */
		template <typename T> 
		void addArrays(T** in_1, T** in_2, T** out, int arrayWidth, int arrayHeight); 
		
		/* Template specialization */
		template void addArrays(int** in_1, int** in_2, int** out, int arrayWidth, int arrayHeight); 
		template void addArrays(float** in_1, float** in_2, float** out, int arrayWidth, int arrayHeight);
		template void addArrays(double** in_1, double** in_2, double** out, int arrayWidth, int arrayHeigh); 

		/* @subArrays */
		template <typename T> 
		void subArrays(T** in_1, T** in_2, T** out, int arrayWidth, int arrayHeight); 

		/* Template specialization */
		template void subArrays(int** in_1, int** in_2, int** out, int arrayWidth, int arrayHeight); 
		template void subArrays(float** in_1, float** in_2, float** out, int arrayWidth, int arrayHeight);
		template void subArrays(double** in_1, double** in_2, double** out, int arrayWidth, int arrayHeight);

		/* @mulArrays */
		template <typename T> 
		void mulArrays(T** in_1, T** in_2, T** out, int arrayWidth, int arrayHeight); 
		
		/* Template specialization */
		template void mulArrays(int** in_1, int** in_2, int** out, int arrayWidth, int arrayHeight); 
		template void mulArrays(float** in_1, float** in_2, float** out, int arrayWidth, int arrayHeight);
		template void mulArrays(double** in_1, double** in_2, double** out, int arrayWidth, int arrayHeight);

		/* @divArrays */
		template <typename T> 
		void divArrays(T** in_1, T** in_2, T** out, int arrayWidth, int arrayHeight); 
		
		/* Template specialization */
		template void divArrays(int** in_1, int** in_2, int** out, int arrayWidth, int arrayHeight); 
		template void divArrays(float** in_1, float** in_2, float** out, int arrayWidth, int arrayHeight);
		template void divArrays(double** in_1, double** in_2, double** out, int arrayWidth, int arrayHeight);
	}
}

int arr_2D::square::get2DIndex(int iElement, int jElement, int arrayWidth)
{
	return (iElement + (jElement * arrayWidth));	
}

template <typename T> 
T** arr_2D::square::createArray(int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	T** array_2D;
	array_2D = (T**) malloc(arrayWidth * sizeof(T*)); 

	for (int i = 0; i < arrayWidth; i++)
		array_2D[i] =  (T*) malloc(arrayHeight * sizeof(T));

	return array_2D; 
}

template <typename T> 
void arr_2D::square::fillArray_SEQ(T** in, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	int ctr = 0;

	for (int j = 0; j < arrayHeight; j++)
		for (int i = 0; i < arrayWidth; i++)
			in[i][j] = (T) ctr++; 
}

template <typename T> 
void arr_2D::square::fillArray_RND(T** in, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	for (int j = 0; j < arrayHeight; j++)
		for (int i = 0; i < arrayWidth; i++)
			in[i][j] = (T) rand(); 
}

template <typename T> 
void arr_2D::square::zeroArray(T** in, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	for (int j = 0; j < arrayHeight; j++)
		for (int i = 0; i < arrayWidth; i++)
			in[i][j] = (T) 0; 
}

template <typename T> 
void arr_2D::square::printArray(T** in, int arrayWidth, int arrayHeight, bool trace)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	std::string resString; 
	std::string traceString = ""; 

	if (trace)
	{
		for (int j = 0; j < arrayHeight; j++)
		{
			for (int i = 0; i < arrayWidth; i++)
			{
				/* @space for "escape sequence" */
				traceString =	util::charArrayToString("@[") + \
								util::intToString(i) + \
								util::charArrayToString("]") + \
								util::charArrayToString("[") + \
								util::intToString(j) + \
								util::charArrayToString("] = ") + \
								util::floatToString(in[i][j]);   
				traceFile::trace(traceString);
			}
		}
	}
	else 
	{
		for (int j = 0; j < arrayHeight; j++)
			for (int i = 0; i < arrayWidth; i++)
				printf("@[%d][%d]= %f \n", i, j, in[i][j]);
	}
}

template <typename T>
T** arr_2D::square::squareArray(T* in, int arrayWidth, int arrayHeight)
{
	T** out; 
	out = arr_2D::square::createArray <T> (arrayWidth, arrayHeight);  

	int index = 0; 

	for (int j = 0; j < arrayHeight; j++)
		for (int i = 0; i < arrayWidth; i++)	
			{
				index = arr_2D::square::get2DIndex(i, j, arrayWidth);

				out[i][j]= in[index]; 
			}

	return out; 
}

template <typename T>
void arr_2D::square::clearArray(T** in, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	for (int i = 0; i < arrayWidth; i++)
	{
		if (in[i])
			free(in[i]);
	}

	if (in)
		free(in);
}

template <typename T> 
void arr_2D::square::addArrays(T** in_1, T** in_2, T** out, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	for (int i = 0; i < arrayWidth ; i++)
		for (int j = 0; j < arrayHeight; j++)
			out[i][j] = in_1[i][j] + in_2[i][j]; 
}

template <typename T> 
void arr_2D::square::subArrays(T** in_1, T** in_2, T** out, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	for (int i = 0; i < arrayWidth ; i++)
		for (int j = 0; j < arrayHeight; j++)
			out[i][j] = in_1[i][j] - in_2[i][j]; 
}

template <typename T> 
void arr_2D::square::mulArrays(T** in_1, T** in_2, T** out, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	for (int i = 0; i < arrayWidth ; i++)
		for (int j = 0; j < arrayHeight; j++)
			out[i][j] = in_1[i][j] * in_2[i][j]; 
}

template <typename T> 
void arr_2D::square::divArrays(T** in_1, T** in_2, T** out, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	for (int i = 0; i < arrayWidth ; i++)
		for (int j = 0; j < arrayHeight; j++)
			out[i][j] = in_1[i][j] / in_2[i][j]; 
}