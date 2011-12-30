#pragma once 

#include "shared.h"

namespace arraySpace
{
	namespace array_1D
	{
		int* createArray(int numItems);
		void fillArray_SEQ(int* arr_1D, int numItems);
		void fillArray_RND(int* arr_1D, int numItems);
		void zeroArray(int* arr_1D, int numItems);

		void addArrays(int* arr_1_1D, int* arr_2_1D, int* outArr_1D, int numItems);
		void subArrays(int* arr_1_1D, int* arr_2_1D, int* outArr_1D, int numItems);
		void mulArrays(int* arr_1_1D, int* arr_2_1D, int* outArr_1D, int numItems);
		void divArrays(int* arr_1_1D, int* arr_2_1D, int* outArr_1D, int numItems);

		void freeArray(int* arr_1D); 
	}
}
 