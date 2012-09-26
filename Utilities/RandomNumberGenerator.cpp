#include "RandomNumberGenerator.h"

template <typename T>
extern
T Utils::GenRandomNumber()
{
    return (T) ((T) rand() / RAND_MAX);
}

template <typename T>
extern
T Utils::GenRandomNumberInRange(int minValue, int maxValue)
{
    return (T) (rand() % (maxValue - minValue) + minValue);
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
