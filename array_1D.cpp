#include "array_1D.h"

int* arraySpace::array_1D::createArray(int numItems)
{
	int* arr_1D; 
	arr_1D = (int*) malloc (sizeof(int*) * numItems); 

	return arr_1D; 
}

void arraySpace::array_1D::fillArray_SEQ(int* arr_1D, int numItems)
{
	for (int i = 0; i < numItems; i++)
		arr_1D[i] = i; 
}

void arraySpace::array_1D::fillArray_RND(int* arr_1D, int numItems)
{
	for (int i = 0; i < numItems; i++)
		arr_1D[i] = rand(); 
}

void arraySpace::array_1D::zeroArray(int* arr_1D, int numItems)
{
	for (int i = 0; i < numItems; i++)
		arr_1D[i] = 0; 
}

void arraySpace::array_1D::addArrays(int* arr_1_1D, int* arr_2_1D, int* outArr_1D, int numItems)
{
	for (int i = 0; i < numItems; i++)
		outArr_1D[i]  = arr_1_1D[i] + arr_2_1D[i]; 
}

void arraySpace::array_1D::subArrays(int* arr_1_1D, int* arr_2_1D, int* outArr_1D, int numItems)
{
	for (int i = 0; i < numItems; i++)
		outArr_1D[i]  = arr_1_1D[i] - arr_2_1D[i]; 
}

void arraySpace::array_1D::mulArrays(int* arr_1_1D, int* arr_2_1D, int* outArr_1D, int numItems)
{
	for (int i = 0; i < numItems; i++)
		outArr_1D[i]  = arr_1_1D[i] * arr_2_1D[i]; 
}

void arraySpace::array_1D::divArrays(int* arr_1_1D, int* arr_2_1D, int* outArr_1D, int numItems)
{
	for (int i = 0; i < numItems; i++)
		outArr_1D[i]  = arr_1_1D[i] + arr_2_1D[i]; 
}

void arraySpace::array_1D::freeArray(int* arr_1D)
{
	if (arr_1D)
		free(arr_1D);  
}