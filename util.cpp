/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: util.cpp
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#include "util.h"

namespace util
{
	int stringToInt(std::string _string);
	float stringToFloat(std::string _string); 
	double stringToDouble(std::string _string); 
	char* stringToCharArray(std::string _string); 
	std::string intToString(int _int); 
	std::string floatToString(float _float); 
	std::string doubleToString(double _double); 
	std::string charArrayToString(char* _char_array); 
}

/* */
int util::stringToInt(std::string _string)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	return atoi(_string.c_str());
}

/* */
float util::stringToFloat(std::string _string)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	return atof(_string.c_str());
}

/* */
double util::stringToDouble(std::string _string)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	return atof(_string.c_str());
}

/* not implemented yet ... */
char* util::stringToCharArray(std::string _string)
{	
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__))); 

	return (char*) _string.c_str();   
}

/* */
std::string util::intToString(int _int)
{ 
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	std::stringstream stStream; 
	stStream << (_int);

	return stStream.str(); 
}

/* */
std::string util::floatToString(float _float)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	std::stringstream stStream; 
	stStream << (_float); 

	return stStream.str(); 
}

/* */
std::string util::doubleToString(double _double)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));
	
	std::stringstream stStream; 
	stStream << (_double); 

	return stStream.str(); 
}

/* */
std::string util::charArrayToString(char* _char_array)
{
	/* @logging */
	logFile::header((std::string(__FILE__) + "\n" + std::string(__FUNCTION__)));

	std::string _string; 
	for (int i = 0; i < strlen(_char_array); i++)
		_string += _char_array[i]; 

	return _string; 
}
