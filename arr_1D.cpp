/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: arr_1D.cpp
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#include "arr_1D.h"

namespace arr_1D
{	
	/* @createArray */
	template <typename T>
	T* createArray(int arrayLenght);

	/* Template specialization */
	template int* createArray(int arrayLenght);
	template float* createArray(int arrayLenght);
	template double* createArray(int arrayLenght);

	/* @fillArray_SEQ */
	template <typename T> 
	void fillArray_SEQ(T* in, int arrayLenght);

	/* Template specialization */
	template void fillArray_SEQ(int* in, int arrayLenght);
	template void fillArray_SEQ(float* in, int arrayLenght);
	template void fillArray_SEQ(double* in, int arrayLenght);

	
	/* @zeroArray */
	template <typename T> 
	void zeroArray(T* in, int arrayLenght);

	/* Template specialization */
	template void zeroArray(int* in, int arrayLenght);
	template void zeroArray(float* in, int arrayLenght);
	template void zeroArray(double* in, int arrayLenght);
	
	/* @printArray */
	template <typename T> 
	void printArray(T* in, int arrayLenght, bool trace);

	/* Template specialization */
	template void printArray(int* in, int arrayLenght, bool trace);
	template void printArray(float* in, int arrayLenght, bool trace);
	template void printArray(double* in, int arrayLenght, bool trace);
	
	/* @clearArray_H */
	template <typename T> 
	void clearArray(T* array_H);

	/* Template specialization */
	template void clearArray(int* array_H);
	template void clearArray(float* array_H);
	template void clearArray(double * array_H);
	
	/* @addArrays */
	template <typename T> 
	void addArrays(T* in_1, T* in_2, T* out, int arrayLenght); 
	
	/* Template specialization */
	template void addArrays(int* in_1, int* in_2, int* out, int arrayLenght); 
	template void addArrays(float* in_1, float* in_2, float* out, int arrayLenght);
	template void addArrays(double* in_1, double* in_2, double* out, int arrayLenght);

	/* @subArrays */
	template <typename T> 
	void subArrays(T* in_1, T* in_2, T* out, int arrayLenght); 
	
	/* Template specialization */
	template void subArrays(int* in_1, int* in_2, int* out, int arrayLenght); 
	template void subArrays(float* in_1, float* in_2, float* out, int arrayLenght);
	template void subArrays(double* in_1, double* in_2, double* out, int arrayLenght);

	/* @mulArrays */
	template <typename T> 
	void mulArrays(T* in_1, T* in_2, T* out, int arrayLenght); 
	
	/* Template specialization */
	template void mulArrays(int* in_1, int* in_2, int* out, int arrayLenght); 
	template void mulArrays(float* in_1, float* in_2, float* out, int arrayLenght);
	template void mulArrays(double* in_1, double* in_2, double* out, int arrayLenght);

	/* @divArrays */
	template <typename T> 
	void divArrays(T* in_1, T* in_2, T* out, int arrayLenght); 
	
	/* Template specialization */
	template void divArrays(int* in_1, int* in_2, int* out, int arrayLenght); 
	template void divArrays(float* in_1, float* in_2, float* out, int arrayLenght);
	template void divArrays(double* in_1, double* in_2, double* out, int arrayLenght);
}

/* */
template <typename T> 
T* arr_1D::createArray(int arrayLenght)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	T* array_1D;
	int dataSize = (arrayLenght * sizeof(T));
	array_1D = (T*) malloc(dataSize); 
	
	return array_1D; 
}

/* */
template <typename T> 
void arr_1D::fillArray_SEQ(T* in, int arrayLenght)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	for (int i = 0; i < arrayLenght; i++)
		in[i] = (T) (i); 
}

/* */
template <typename T> 
void arr_1D::zeroArray(T* in, int arrayLenght)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	for (int i = 0; i < arrayLenght; i++)
		in[i] = (T) (0); 
}

/* */
template <typename T> 
void arr_1D::printArray(T* in, int arrayLenght, bool trace)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	std::string resString; 
	std::string traceString = ""; 
	if (trace)
	{
		for (int i = 0; i < arrayLenght; i++)
		{
			 
			/* @space for "escape sequence" */
			traceString =	util::charArrayToString("@[") + \
							util::intToString(i) + \
							util::charArrayToString("] = ") + \
							util::floatToString(in[i]);   
			traceFile::trace(traceString);  
		}
	}
	else 
	{
		for (int i = 0; i < arrayLenght; i++)
			printf("@[%d] = %f \n", i, in[i]); 
	}
}

/* */
template <typename T> 
void arr_1D::clearArray(T* array_H)
{	
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	if (array_H)
        free(array_H);
}	

/* */
template <typename T> 
void arr_1D::addArrays(T* in_1, T* in_2, T* out, int arrayLenght)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	for (int i = 0; i < arrayLenght; i++)
		out[i] = in_1[i] + in_2[i];  
}

/* */
template <typename T> 
void arr_1D::subArrays(T* in_1, T* in_2, T* out, int arrayLenght)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	for (int i = 0; i < arrayLenght; i++)
		out[i] = in_1[i] - in_2[i];  
}

/* */
template <typename T> 
void arr_1D::mulArrays(T* in_1, T* in_2, T* out, int arrayLenght)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	for (int i = 0; i < arrayLenght; i++)
		out[i] = in_1[i] * in_2[i];  
}

/* */
template <typename T> 
void arr_1D::divArrays(T* in_1, T* in_2, T* out, int arrayLenght)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	for (int i = 0; i < arrayLenght; i++)
		out[i] = in_1[i] / in_2[i];  
}
