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

#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <typeinfo>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <stdio.h>
#include <string.h>

#include "Typedefs.h"
#include "Globals.h"

namespace Utils
{
    template <typename T>
    extern
    T StringToNum(string_t string);

    template
    char StringToNum <char> (string_t string);

    template
    unsigned char StringToNum <unsigned char> (string_t string);

    template
    int StringToNum <int> (string_t string);

    template
    unsigned int StringToNum <unsigned int> (string_t string);

    template
    float StringToNum <float> (string_t string);

    template
    double StringToNum <double> (string_t string);

    template <typename T>
    extern
    string_t NumToString(T value);

    template
    string_t NumToString <char> (char value);

    template
    string_t NumToString <unsigned char> (unsigned char value);

    template
    string_t NumToString <int> (int value);

    template
    string_t NumToString <unsigned int> (unsigned int value);

    template
    string_t NumToString <float> (float value);

    template
    string_t NumToString <double> (double value);

    template <typename T>
    extern
    T GenRandomNumber();

    template
    char GenRandomNumber <char> ();

    template
    unsigned char GenRandomNumber <unsigned char> ();

    template
    int GenRandomNumber <int> ();

    template
    unsigned int GenRandomNumber <unsigned int> ();

    template
    float GenRandomNumber <float> ();

    template
    double GenRandomNumber <double> ();

    template <typename T>
    extern
    T GenRandomNumberInRange(int minValue, int maxValue);

    template
    char GenRandomNumberInRange <char>
    (int minValue, int maxValue);

    template
    unsigned char GenRandomNumberInRange <unsigned char>
    (int minValue, int maxValue);

    template
    int GenRandomNumberInRange <int>
    (int minValue, int maxValue);

    template
    unsigned int GenRandomNumberInRange <unsigned int>
    (int minValue, int maxValue);

    template
    float GenRandomNumberInRange <float>
    (int minValue, int maxValue);

    template
    double GenRandomNumberInRange <double>
    (int minValue, int maxValue);


    int stringToInt(string_t string);
    int stringToInt_const(const string_t string);
    float stringToFloat(string_t string);
    float stringToFloat_const(const string_t string);
    double stringToDouble(string_t string);
    double stringToDouble_const(const string_t string);

	string_t intToString(int intVal);
    string_t intToString_const(const int intVal);
	string_t floatToString(float floatVal);
    string_t floatToString_const(const float floatVal);
	string_t doubleToString(double doubleVal);
    string_t doubleToString_const(const double doubleVal);
	string_t charArrayToString(char* inputCharArray);
    string_t charArrayToString_const(const char* inputCharArray);



    char* stringToCharArray(string_t inputString);

    int rand_int();
    float rand_float();
    double rand_double();
    int rand_int_range(int minNum, int maxNum);

}

#endif /* _UTILS_H_ */
