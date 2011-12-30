/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: arr_1D.h
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#pragma once

#include "cuShared.h"
#include "logFile.h"
#include "traceFile.h"
#include "util.h"

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


	void transposeArray_FLAT();
	void transposeArray_SQUARE(); 
}
