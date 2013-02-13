#ifndef _MEMORY_MACROS_H_
#define _MEMORY_MACROS_H_

#include <iostream>
#include <fstream>

#include "Utilities/Utils.h"
#include "Memory/Memory_1D.h"
#include "Memory/Memory_2D.h"
#include "Memory/Memory_3D.h"

#include "Memory/Vector.h"
#include "Memory/Image.h"
#include "Memory/FlatImage.h"
#include "Memory/Volume.h"
#include "Memory/FlatVolume.h"

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


#define MEM_ALLOC_VECTOR ( TYPE, NX )               ( new Yuri::Memory::Vector <TYPE> ( NX ) )
#define MEM_ALLOC_IMAGE ( TYPE, NX , NY )           ( new Yuri::Memory::Image <TYPE> ( NX, NY ) )
#define MEM_ALLOC_VOLUME ( TYPE, NX , NY, NZ )      ( new Yuri::Memory::Volume <TYPE> ( NX, NY, NZ ) )
#define MEM_ALLOC_FLAT_IMAGE ( TYPE, NX , NY )      ( new Yuri::Memory::FlatImage <TYPE> ( NX, NY ) )
#define MEM_ALLOC_FLAT_VOLUME ( TYPE, NX , NY, NZ ) ( new Yuri::Memory::FlatVolume <TYPE> ( NX, NY, NZ ) )

#define FREE_VECTOR ( PTR )       ( delete ( PTR ) )
#define FREE_IMAGE ( PTR )        ( delete ( PTR ) )
#define FREE_VOLUME ( PTR )       ( delete ( PTR ) )
#define FREE_FLAT_IMAGE ( PTR )   ( delete ( PTR ) )
#define FREE_FLAT_VOLUME ( PTR )  ( delete ( PTR ) )

#endif // _MEMORY_MACROS_H_
