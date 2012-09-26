#ifndef _TYPE_CONVERSION_H_
#define _TYPE_CONVERSION_H_

namespace Utils
{
template <typename T>
extern
T StringToNum(string_t string);

template <typename T>
extern
string_t NumToString(T value);

template
char StringToNum <char>
(string_t string);

template
unsigned char StringToNum <unsigned char>
(string_t string);

template
int StringToNum <int>
(string_t string);

template
unsigned int StringToNum <unsigned int>
(string_t string);

template
float StringToNum <float>
(string_t string);

template
double StringToNum <double>
(string_t string);

template
string_t NumToString <char>
(char value);

template
string_t NumToString <unsigned char>
(unsigned char value);

template
string_t NumToString <int>
(int value);

template
string_t NumToString <unsigned int>
(unsigned int value);

template
string_t NumToString <float>
(float value);

template
string_t NumToString <double>
(double value);
}

#endif // _TYPE_CONVERSION_H_
