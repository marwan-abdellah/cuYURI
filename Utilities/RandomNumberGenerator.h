#ifndef _RANDOM_NUMBER_GENERATOR_H_
#define _RANDOM_NUMBER_GENERATOR_H_

namespace Utils
{
template <typename T>
extern
T GenRandomNumber();

template <typename T>
extern
T GenRandomNumberInRange
(int minValue, int maxValue);

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
}

#endif // _RANDOM_NUMBER_GENERATOR_H_
