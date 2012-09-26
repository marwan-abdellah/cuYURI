#include "TypeConversion.h"

template <typename T>
extern
T Utils::StringToNum(string_t string)
{
    return (T) atoi(string.c_str());
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

template
string_t Utils::NumToString <char>
(char value);

template
string_t Utils::NumToString <unsigned char>
(unsigned char value);

template
string_t Utils::NumToString <int>
(int value);

template
string_t Utils::NumToString <unsigned int>
(unsigned int value);

template
string_t Utils::NumToString <float>
(float value);

template
string_t Utils::NumToString <double>
(double value);
