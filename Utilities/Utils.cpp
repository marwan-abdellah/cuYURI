/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File			: Volume
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created		: April 2011
 * Description	:
 * Note(s)		:
 *********************************************************************/

#include "Utils.h"


template <typename T>
extern
T Utils::StringToNum(string_t string)
{
    return (T) atoi(string.c_str());
}

template
char Utils::StringToNum <char> (string_t string);

template
unsigned char Utils::StringToNum <unsigned char> (string_t string);

template
int Utils::StringToNum <int> (string_t string);

template
unsigned int Utils::StringToNum <unsigned int> (string_t string);

template
float Utils::StringToNum <float> (string_t string);

template
double Utils::StringToNum <double> (string_t string);



int Utils::stringToInt(string_t string)
{
	return atoi(string.c_str());
}

float Utils::stringToFloat(string_t string)
{
	return atof(string.c_str());
}

double Utils::stringToDouble(string_t string)
{
	return atof(string.c_str());
}

template <typename T>
extern
string_t Utils::NumToString(T value)
{
    sstream_t stream;
    stream << ((T) value);
    return stream.str();
}

template
string_t Utils::NumToString <char> (char value);

template
string_t Utils::NumToString <unsigned char> (unsigned char value);

template
string_t Utils::NumToString <int> (int value);

template
string_t Utils::NumToString <unsigned int> (unsigned int value);

template
string_t Utils::NumToString <float> (float value);

template
string_t Utils::NumToString <double> (double value);



string_t Utils::intToString(int intVal)
{
	sstream_t stream;
	stream << (intVal);
	return stream.str();
}

string_t Utils::floatToString(float floatVal)
{
	sstream_t stream;
	stream << (floatVal);
	return stream.str();
}

string_t Utils::doubleToString(double doubleVal)
{
	sstream_t stream;
	stream << (doubleVal);
	return stream.str();
}

string_t Utils::charArrayToString(char* inputCharArray)
{
	string_t string;
	for (int i = 0; i < strlen(inputCharArray); i++)
		string += inputCharArray[i];

	return string;
}

char* Utils::stringToCharArray(string_t inputString)
{

	int stringLen = inputString.size();

	char* outputChar;
	for (int i = 0; i <= stringLen ; i++)
	{
		outputChar[i] = inputString[i];
	}

	return outputChar;
}

string_t Utils::charArrayToString_const (const char* inputCharArray)
{
    string_t string;
    for (int i = 0; i < strlen(inputCharArray); i++)
        string += inputCharArray[i];

    return string;
}





template <typename T>
extern
T Utils::GenRandomNumber()
{
    return (T) ((T) rand() / RAND_MAX);
}

template
char Utils::GenRandomNumber <char> ();

template
unsigned char Utils::GenRandomNumber <unsigned char> ();

template
int Utils::GenRandomNumber <int> ();

template
unsigned int Utils::GenRandomNumber <unsigned int> ();

template
float Utils::GenRandomNumber <float> ();

template
double Utils::GenRandomNumber <double> ();

template <typename T>
extern
T Utils::GenRandomNumberInRange(int minValue, int maxValue)
{
    return (T) (rand() % (maxValue - minValue) + minValue);
}

template
char Utils::GenRandomNumberInRange <char>
(int minValue, int maxValue);

template
unsigned char Utils::GenRandomNumberInRange <unsigned char>
(int minValue, int maxValue);

template
int Utils::GenRandomNumberInRange <int>
(int minValue, int maxValue);

template
unsigned int Utils::GenRandomNumberInRange <unsigned int>
(int minValue, int maxValue);

template
float Utils::GenRandomNumberInRange <float>
(int minValue, int maxValue);

template
double Utils::GenRandomNumberInRange <double>
(int minValue, int maxValue);


int Utils::rand_int()
{
	return rand();
}

float Utils::rand_float()
{
	return (float) float(rand()) / RAND_MAX;
}

double Utils::rand_double()
{
	return (double) (double(drand48()) / RAND_MAX);
}

int Utils::rand_int_range(int minNum, int maxNum)
{
	 return( rand() % (maxNum - minNum) + minNum);
}
