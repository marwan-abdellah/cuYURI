/*
***************************************************************************
	Author(s)		: Marwan Abdellah
	Email(s)		: <abdellah.marwan@gmail.com>
	Module			: util.h
	Description		: 
	Created			: 
	Note(s)			:
***************************************************************************
*/

#pragma once 

#include "cuShared.h"
#include "logFile.h"
#include "traceFile.h"


namespace util
{
	/* @stringToInt */
	int stringToInt(std::string _string);

	/* @stringToFloat */
	float stringToFloat(std::string _string); 

	/* @stringToDouble */
	double stringToDouble(std::string _string); 

	/* (char*) _string.c_str() */
	char* stringToCharArray(std::string _string); 
	
	/* @intToString */
	std::string intToString(int _int); 

	/* @floatToString */
	std::string floatToString(float _float); 

	/* @doubleToString */
	std::string doubleToString(double _double); 

	/* @charArrayToString */
	std::string charArrayToString(char* _char_array); 
}