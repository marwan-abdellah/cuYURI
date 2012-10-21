#ifndef _MEMORY_MACROS_H_
#define _MEMORY_MACROS_H_

#include <iostream>
#include <fstream>

#include "Utilities/Utils.h"
#include "Memory/Memory_1D.h"
#include "Memory/Memory_2D.h"
#include "Memory/Memory_3D.h"

/*!
 * Generic allocation of 1D arrays.
 */
#define MEM_ALLOC_1D_GENRIC( TYPE, NX )                             \
    ( TYPE* malloc ( sizeof( TYPE ) * NX ) )

/*!
 * Customized allocation of 1D arrays.
 */
#define MEM_ALLOC_1D( TYPE, NX )                                    \
    ( Memory::Alloc_1D_Array < TYPE > ( NX ) )

/*!
 * Customized allocation of 2D arrays.
 */
#define MEM_ALLOC_2D( TYPE, NX, NY )                                \
    ( Memory::Alloc_2D_Array < TYPE > ( NX, NY ) )

/*!
 * Customized allocation of 3D arrays.
 */
#define MEM_ALLOC_3D( TYPE, NX, NY, NZ )                            \
    ( Memory::Alloc_3D_Array < TYPE > ( NX, NY , NZ ) )

/*!
 * Customized deallocation of 1D arrays.
 */
#define FREE_MEM_1D( PTR, TYPE )                                    \
    ( Memory::Free_1D_Array < TYPE > (PTR) )

/*!
 * Customized deallocation of 2D arrays.
 */
#define FREE_MEM_2D ( PTR, NX, NY, TYPE )                           \
        ( Memory::Free_2D_Array < TYPE > ( PTR, NX, NY ) )

/*!
 * Customized deallocation of 3D arrays.
 */
#define FREE_MEM_3D ( PTR, NX, NY, NZ, TYPE )                       \
        ( Memory::Free_3D_Array < TYPE > ( PTR, NX, NY, NZ ) )

#endif // _MEMORY_MACROS_H_
