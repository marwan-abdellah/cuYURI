/*********************************************************************
 * Copyrights (c) Marwan Abdellah. All rights reserved.
 * This code is part of my Master's Thesis Project entitled "High
 * Performance Fourier Volume Rendering on Graphics Processing Units
 * (GPUs)" and submitted to the Systems & Biomedical Engineering
 * Department, Faculty of Engineering, Cairo University.
 * Please, don't use or distribute without authors' permission.

 * File			: MACROS.h
 * Author(s)	: Marwan Abdellah <abdellah.marwan@gmail.com>
 * Created		: April 2011
 * Description	:
 * Note(s)		:
 *********************************************************************/

#ifndef _MACROS_H_
#define _MACROS_H_

#include <iostream>
#include <fstream>
#include "Utilities/Typedefs.h"
#include "Utilities/Utils.h"


#define PRINT_INFO 1

/* @ Utilities */
#define ITS( INT ) ( Utils::intToString( INT ) )
#define FTS( FLT ) ( Utils::floatToString( FLT ) )
#define DTS( DBL ) ( Utils::doubleToString( DBL ) )
#define CITS( INT ) ( Utils::intToString_const( INT ) )
#define CATS( CHAR ) ( Utils::charArrayToString( CHAR ) )
#define CCATS( CHAR ) ( Utils::charArrayToString_const( CHAR ) )
#define STI( INT ) ( Utils::stringToInt( INT ) )
#define CSTI( INT ) ( stringToInt_const( INT ) )
#define STF( FLOAT ) ( stringToFloat( FLOAT ) )
#define CSTF( FLOAT ) ( stringToFloat_const( FLOAT ))
#define STD( DOUBLE ) ( stringToDouble( DOUBLE ) )
#define CSTD( DOUBLE ) ( stringToDouble_const( DOUBLE ) )
#define STCA ( STRNG ) ( Utils::stringToCharArray( STRNG ) )

/* @ Messaging & Logging */
#ifdef PRINT_INFO
    #define INFO( MESSAGE )                                         			\
    COUT << STRG( __FILE__ ) << ":[" << ( __LINE__ ) << "]" <<      			\
    ENDL << TAB << "* " << STRG( __FUNCTION__ ) << TAB <<          				\
    STRG( MESSAGE ) << ENDL;
#else
    #define INFO(MESSAGE) /* PRINT NOTHING */;
#endif

#define SEP( MESSAGE )															\
    COUT << 																	\
    "********************************************************"					\
    << ENDL;

#define INFO_LOOP( ITERATION, MESSAGE )    \
    COUT << TAB << "@[" << ( ITERATION ) << "] -> " <<                          \
    STRG( MESSAGE ) << ENDL;



#ifdef PRINT_DEBUG_INFO
    #define DBG_INFO( MESSAGE )                                     			\
    COUT << STRG( __FILE__ ) << ":" << ( __LINE__ ) << ":" <<					\
    ENDL << "[" << STRG( __DATE__ ) << "]" << ":" << 							\
    "[" << STRG( __TIME__ ) << "]" <<	TAB <<									\
    STRG( __FUNCTION__ ) << TAB << STRG( MESSAGE ) << ENDL;
#else
    #define DBG_INFO( MESSAGE ) /* PRINT NOTHING */;
#endif

#define EXIT( CODE ) 															\
    COUT << STRG( __FILE__ ) << ":[" <<                             			\
    ( __LINE__ ) << "]" << ENDL << TAB <<                           			\
    STRG( __FUNCTION__ ) << ": EXITING " << CODE << ENDL;           			\
    exit( 0 );

/* @ Memory Allocation */
#define MEM_ALLOC_1D_GENERIC(TYPE, SIZE) 										\
		((TYPE*)  malloc (SIZE * sizeof(TYPE)))
#define MEM_ALLOC_1D(TYPE, SIZE_X)                                              \
		(Memory::alloc_1D <TYPE> (SIZE_X))
#define MEM_ALLOC_2D( TYPE, SIZE_X, SIZE_Y )                                    \
		(Memory::alloc_2D <TYPE> (SIZE_X, SIZE_Y))
#define MEM_ALLOC_3D( TYPE, SIZE_X, SIZE_Y, SIZE_Z)                             \
		(Memory::alloc_3D <TYPE> (SIZE_X, SIZE_Y , SIZE_Z))


#define DISPLAY_KERNEL_PROFILING_DATA (KERNEL_NAME, PROFILE)                    \
    SEP();                                                                      \
    COUT << TAB << "Kernel Execution Time @[" << KERNEL_NAME " << "]" << ENDL;         \
    COUT << TAB << TAB << "@Nano-Seconds (ns)  : " << (PROFILE.kernelDuration * (1000 * 1000)) << ENDL;                   \
    COUT << TAB << TAB << "@Micro-Seconds (us) : " << (PROFILE.kernelDuration * 1000) << ENDL;                            \
    COUT << TAB << TAB << "@Milli-Seconds (ms) : " <<  PROFILE.kernelDuration << ENDL;                                     \
    COUT << TAB << TAB << "@Seconds (s)        : " << (PROFILE.kernelDuration / (1000)) << ENDL;                            \
    SEP();

#endif /* _MACROS_H_ */
