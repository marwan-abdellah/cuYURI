/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: arr_2D_FLAT.h
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

namespace arr_2D
{
	namespace flat
	{
		int get2DIndex(int iElement, int jElement, int arrayWidth);

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