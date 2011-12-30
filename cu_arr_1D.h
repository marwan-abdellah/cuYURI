/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: cu_arr_1D.h
	Description		:
	Created			: 
	Note(s)			:
***************************************************************************
*/

#pragma once

#include "cuShared.h"
#include "logFile.h"

namespace CU
{
	namespace arr_1D
	{
		/* @createArray_D */
		template <typename T>
		T* createArray_D(int arrayLenght);

		/* Template specialization */
		template int* createArray_D(int arrayLenght);
		template float* createArray_D(int arrayLenght);
		template double* createArray_D(int arrayLenght);

		/* @createArray_H */
		template <typename T>
		T* createArray_H(int arrayLenght);

		/* Template specialization */
		template int* createArray_H(int arrayLenght);
		template float* createArray_H(int arrayLenght);
		template double* createArray_H(int arrayLenght);

		/* @fillArray_SEQ_H */
		template <typename T>
		void fillArray_SEQ_H(T* in, int arrayLenght);

		/* Template specialization */
		template void fillArray_SEQ_H(int* in, int arrayLenght);
		template void fillArray_SEQ_H(float* in, int arrayLenght);
		template void fillArray_SEQ_H(double* in, int arrayLenght);

		/* @uploadData */
		template <typename T>
		void uploadData(T* array_H, T* array_D, int arraySize); 

		/* Template specialization */
		template void uploadData(int* array_H, int* array_D, int arraySize);
		template void uploadData(float* array_H, float* array_D, int arraySize);
		template void uploadData(double* array_H, double* array_D, int arraySize);

		/* @downloadData */
		template <typename T>
		void downloadData(T* array_H, T* array_D, int arraySize);

		/* Template specialization */
		template void downloadData(int* array_H, int* array_D, int arraySize);
		template void downloadData(float* array_H, float* array_D, int arraySize);
		template void downloadData(double* array_H, double* array_D, int arraySize);

		/* @clearArray_D */
		template <typename T>
		void clearArray_D(T* array_D);

		/* Template specialization */
		template void clearArray_D(int* array_D);
		template void clearArray_D(float* array_D);
		template void clearArray_D(double* array_D);

		/* @clearArray_H */
		template <typename T>
		void clearArray_H(T* array_H); 

		/* Template specialization */
		template void clearArray_H(int* array_H); 
		template void clearArray_H(float* array_H); 
		template void clearArray_H(double* array_H); 

		/* @cuAddArrays */
		template <typename T>
		void cuAddArrays(T* in_1, T* in_2, T* out, int arraySize, gpuConfig_1D* conf); 
	
		/* Template specialization */
		template void cuAddArrays(int* in_1, int* in_2, int* out, int arraySize, gpuConfig_1D* conf); 
		template void cuAddArrays(float* in_1, float* in_2, float* out, int arraySize, gpuConfig_1D* conf); 
		template void cuAddArrays(double* in_1, double* in_2, double* out, int arraySize, gpuConfig_1D* conf); 

		void cuAddArrays_i(int* in_1, int* in_2, int* out, int arraySize, gpuConfig_1D* conf);
		void cuAddArrays_f(float* in_1, float* in_2, float* out, int arraySize, gpuConfig_1D* conf);
		void cuAddArrays_d(double* in_1, double* in_2, double* out, int arraySize, gpuConfig_1D* conf);

		extern "C" void add_i_1D(int* in_1, int* in_2, int* out, int arraySize, gpuConfig_1D* conf); 
		extern "C" void add_f_1D(float* in_1, float* in_2, float* out, int arraySize, gpuConfig_1D* conf);
		extern "C" void add_d_1D(double* in_1, double* in_2, double* out, int arraySize, gpuConfig_1D* conf);

		void cuMulArrays_i(int* in_1, int* in_2, int* out, int arraySize, gpuConfig_1D* conf);
		void cuMulArrays_f(float* in_1, float* in_2, float* out, int arraySize, gpuConfig_1D* conf);
		void cuMulArrays_d(double* in_1, double* in_2, double* out, int arraySize, gpuConfig_1D* conf);

		extern "C" void mul_i_1D(int* in_1, int* in_2, int* out, int arraySize, gpuConfig_1D* conf); 
		extern "C" void mul_f_1D(float* in_1, float* in_2, float* out, int arraySize, gpuConfig_1D* conf);
		extern "C" void mul_d_1D(double* in_1, double* in_2, double* out, int arraySize, gpuConfig_1D* conf);

		void cuSubArrays_i(int* in_1, int* in_2, int* out, int arraySize, gpuConfig_1D* conf);
		void cuSubArrays_f(float* in_1, float* in_2, float* out, int arraySize, gpuConfig_1D* conf);
		void cuSubArrays_d(double* in_1, double* in_2, double* out, int arraySize, gpuConfig_1D* conf);
		
		extern "C" void sub_i_1D(int* in_1, int* in_2, int* out, int arraySize, gpuConfig_1D* conf); 
		extern "C" void sub_f_1D(float* in_1, float* in_2, float* out, int arraySize, gpuConfig_1D* conf);
		extern "C" void sub_d_1D(double* in_1, double* in_2, double* out, int arraySize, gpuConfig_1D* conf);

		void cuDivArrays_i(int* in_1, int* in_2, int* out, int arraySize, gpuConfig_1D* conf);
		void cuDivArrays_f(float* in_1, float* in_2, float* out, int arraySize, gpuConfig_1D* conf);
		void cuDivArrays_d(double* in_1, double* in_2, double* out, int arraySize, gpuConfig_1D* conf);

		extern "C" void div_i_1D(int* in_1, int* in_2, int* out, int arraySize, gpuConfig_1D* conf); 
		extern "C" void div_f_1D(float* in_1, float* in_2, float* out, int arraySize, gpuConfig_1D* conf);
		extern "C" void div_d_1D(double* in_1, double* in_2, double* out, int arraySize, gpuConfig_1D* conf);
	}
}
