#include "Test_2D_SQUARE.h"

#define TYPE float

namespace square_2D_Test
{
	char* appName = "add_1D";

	void runTest(int argc, char** argv);
}

void square_2D_Test::runTest(int argc, char** argv)
{
	float** arrPtr1;
	float** arrPtr2; 

	int mWidth = 4; 
	int mHeight = 4; 

	/* Create arrays*/
	arrPtr1 = arr_2D::square::createArray <TYPE> (mWidth, mHeight); 
	arrPtr2 = arr_2D::square::createArray <TYPE> (mWidth, mHeight);

	/* Fill arrays */
	arr_2D::square::fillArray_SEQ <TYPE> (arrPtr1, mWidth, mHeight);
	arr_2D::square::fillArray_RND <TYPE> (arrPtr2, mWidth, mHeight); 

	/* Print arrays */
	arr_2D::square::printArray <TYPE> (arrPtr1, mWidth, mHeight, false);
	arr_2D::square::printArray <TYPE> (arrPtr2, mWidth, mHeight, false);


}
