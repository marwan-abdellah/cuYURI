/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: cu_arr_2D_GM_GLOBAL.h
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
	namespace arr_2D
	{
		namespace globalMem
		{
			/* @getPitch */
			template <typename T>
			pitchSize getPitch(int arrayWidth, int arrayHeight, T __NULL__);

			/* Template specialization */
			template pitchSize getPitch(int arrayWidth, int arrayHeight, int __NULL__);
			template pitchSize getPitch(int arrayWidth, int arrayHeight, float __NULL__);
			template pitchSize getPitch(int arrayWidth, int arrayHeight, double __NULL__);

			/* @createArray_D */
			template <typename T>
			T* createArray_D(int arrayWidth, int arrayHeight);

			/* Template specialization */
			template int* createArray_D(int arrayWidth, int arrayHeight);
			template float* createArray_D(int arrayWidth, int arrayHeight);
			template double* createArray_D(int arrayWidth, int arrayHeight);

			/* @createArray_H */
			template <typename T>
			T* createArray_H(int arrayWidth, int arrayHeight);

			/* Template specialization */
			template int* createArray_H(int arrayWidth, int arrayHeight);
			template float* createArray_H(int arrayWidth, int arrayHeight);
			template double* createArray_H(int arrayWidth, int arrayHeight);

			/* @fillArray_SEQ_H */
			template <typename T>
			void fillArray_SEQ_H(T* in, int arrayWidth, int arrayHeight);

			/* Template specialization */
			template void fillArray_SEQ_H(int* in, int arrayWidth, int arrayHeight);
			template void fillArray_SEQ_H(float* in, int arrayWidth, int arrayHeight);
			template void fillArray_SEQ_H(double* in, int arrayWidth, int arrayHeight);

			/* @uploadData */
			template <typename T>
			void uploadData(T* array_H, T* array_D, pitchSize __pitch, int arrayWidth, int arrayHeight); 

			/* Template specialization */
			template void uploadData(int* array_H, int* array_D, pitchSize __pitch, int arrayWidth, int arrayHeight);
			template void uploadData(float* array_H, float* array_D, pitchSize __pitch, int arrayWidth, int arrayHeight);
			template void uploadData(double* array_H, double* array_D, pitchSize __pitch, int arrayWidth, int arrayHeight);

			/* @downloadData */
			template <typename T>
			void downloadData(T* array_H, T* array_D, pitchSize __pitch, int arrayWidth, int arrayHeight);

			/* Template specialization */
			template void downloadData(int* array_H, int* array_D, pitchSize __pitch, int arrayWidth, int arrayHeight);
			template void downloadData(float* array_H, float* array_D, pitchSize __pitch, int arrayWidth, int arrayHeight);
			template void downloadData(double* array_H, double* array_D, pitchSize __pitch, int arrayWidth, int arrayHeight);

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

			void cuAddArrays_i(int* in_1, int* in_2, int* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf); 
			void cuAddArrays_f(float* in_1, float* in_2, float* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf); 
			void cuAddArrays_d(double* in_1, double* in_2, double* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);

			extern "C" void add_i_2D_GM(int* in_1, int* in_2, int* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf); 
			extern "C" void add_f_2D_GM(float* in_1, float* in_2, float* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);
			extern "C" void add_d_2D_GM(double* in_1, double* in_2, double* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);

			void cuMulArrays_i(int* in_1, int* in_2, int* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);
			void cuMulArrays_f(float* in_1, float* in_2, float* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);
			void cuMulArrays_d(double* in_1, double* in_2, double* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);

			extern "C" void mul_i_2D_GM(int* in_1, int* in_2, int* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf); 
			extern "C" void mul_f_2D_GM(float* in_1, float* in_2, float* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);
			extern "C" void mul_d_2D_GM(double* in_1, double* in_2, double* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);

			void cuSubArrays_i(int* in_1, int* in_2, int* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);
			void cuSubArrays_f(float* in_1, float* in_2, float* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);
			void cuSubArrays_d(double* in_1, double* in_2, double* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);
			
			extern "C" void sub_i_2D_GM(int* in_1, int* in_2, int* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf); 
			extern "C" void sub_f_2D_GM(float* in_1, float* in_2, float* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);
			extern "C" void sub_d_2D_GM(double* in_1, double* in_2, double* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);

			void cuDivArrays_i(int* in_1, int* in_2, int* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);
			void cuDivArrays_f(float* in_1, float* in_2, float* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);
			void cuDivArrays_d(double* in_1, double* in_2, double* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);

			extern "C" void div_i_2D_GM(int* in_1, int* in_2, int* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf); 
			extern "C" void div_f_2D_GM(float* in_1, float* in_2, float* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);
			extern "C" void div_d_2D_GM(double* in_1, double* in_2, double* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf);
		}
	}
}