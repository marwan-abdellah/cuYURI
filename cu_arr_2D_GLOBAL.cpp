/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: cu_arr_2D_GM_GLOBAL.cpp
	Description		:
	Created			: 
	Note(s)			:
***************************************************************************
*/

#include "cu_arr_2D_GLOBAL.h"

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

template <typename T>
pitchSize CU::arr_2D::globalMem::getPitch(int arrayWidth, int arrayHeight, T __NULL__)
{
	float* array_2D_GM;
	pitchSize __pitch; 
	
	/* Just do this allocation to get the pitch and then release the array */
	cudaMallocPitch((void**)&array_2D_GM, &__pitch, arrayHeight * sizeof(T), arrayWidth);
	cudaFree(array_2D_GM); 

	return __pitch;
}

template <typename T>
T* CU::arr_2D::globalMem::createArray_D(int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	T* array_2D_GM;
	pitchSize __pitch;
	cudaMallocPitch((void**)&array_2D_GM, &__pitch, arrayHeight * sizeof(float), arrayWidth);

	return array_2D_GM; 
}

template <typename T>
T* CU::arr_2D::globalMem::createArray_H(int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	T* array_2D_GM;
	int dataSize = ((arrayWidth * arrayHeight) * sizeof(T));
	array_2D_GM = (T*) malloc(dataSize); 
	
	return array_2D_GM; 
}

template <typename T>
void CU::arr_2D::globalMem::uploadData(T* array_H, T* array_D, pitchSize __pitch, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));
	
	cudaMemcpy2D(array_D, __pitch, array_H, arrayHeight * sizeof(T), arrayHeight * sizeof(T), arrayWidth, cudaMemcpyHostToDevice);
}

template <typename T>
void CU::arr_2D::globalMem::downloadData(T* array_H, T* array_D, pitchSize __pitch, int arrayWidth, int arrayHeight)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	cudaMemcpy2D(array_H, arrayHeight * sizeof(T), array_D, __pitch, arrayHeight * sizeof(T), arrayWidth, cudaMemcpyDeviceToHost);
}

template <typename T>
void CU::arr_2D::globalMem::clearArray_D(T* array_D)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	if (array_D)
        cudaFree(array_D);
}

template <typename T>
void CU::arr_2D::globalMem::clearArray_H(T* array_H)
{	
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	if (array_H)
        free(array_H);
}

void CU::arr_2D::globalMem::cuAddArrays_i(int* in_1, int* in_2, int* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	add_i_2D_GM(in_1, in_2, out, arrayWidth, arrayHeight, __pitch, conf);
}

void CU::arr_2D::globalMem::cuAddArrays_f(float* in_1, float* in_2, float* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	add_f_2D_GM(in_1, in_2, out, arrayWidth, arrayHeight, __pitch, conf);
}

void CU::arr_2D::globalMem::cuAddArrays_d(double* in_1, double* in_2, double* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf)
{	
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));
	
	add_d_2D_GM(in_1, in_2, out, arrayWidth, arrayHeight, __pitch, conf);
}

void CU::arr_2D::globalMem::cuMulArrays_i(int* in_1, int* in_2, int* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	mul_i_2D_GM(in_1, in_2, out, arrayWidth, arrayHeight, __pitch, conf);
}

void CU::arr_2D::globalMem::cuMulArrays_f(float* in_1, float* in_2, float* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	mul_f_2D_GM(in_1, in_2, out, arrayWidth, arrayHeight, __pitch, conf);
}

void CU::arr_2D::globalMem::cuMulArrays_d(double* in_1, double* in_2, double* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf)
{	
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));
	
	mul_d_2D_GM(in_1, in_2, out, arrayWidth, arrayHeight, __pitch, conf);
}

void CU::arr_2D::globalMem::cuSubArrays_i(int* in_1, int* in_2, int* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	sub_i_2D_GM(in_1, in_2, out, arrayWidth, arrayHeight, __pitch, conf);
}

void CU::arr_2D::globalMem::cuSubArrays_f(float* in_1, float* in_2, float* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	sub_f_2D_GM(in_1, in_2, out, arrayWidth, arrayHeight, __pitch, conf);
}

void CU::arr_2D::globalMem::cuSubArrays_d(double* in_1, double* in_2, double* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf)
{	
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));
	
	sub_d_2D_GM(in_1, in_2, out, arrayWidth, arrayHeight, __pitch, conf);
}

void CU::arr_2D::globalMem::cuDivArrays_i(int* in_1, int* in_2, int* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	div_i_2D_GM(in_1, in_2, out, arrayWidth, arrayHeight, __pitch, conf);
}

void CU::arr_2D::globalMem::cuDivArrays_f(float* in_1, float* in_2, float* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	div_f_2D_GM(in_1, in_2, out, arrayWidth, arrayHeight, __pitch, conf);
}

void CU::arr_2D::globalMem::cuDivArrays_d(double* in_1, double* in_2, double* out, pitchSize __pitch, int arrayWidth, int arrayHeight, gpuConfig_XD* conf)
{	
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));
	
	div_d_2D_GM(in_1, in_2, out, arrayWidth, arrayHeight, __pitch, conf);
}

