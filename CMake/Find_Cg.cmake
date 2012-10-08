#####################################################################
# Copyright © 2011-2012,
# Marwan Abdellah: <abdellah.marwan@gmail.com>
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA 02110-1301, USA.
#####################################################################

#####################################################################
# Try to find NVIDIA's Cg compiler, runtime libraries, and
# include path.
#
# Defined Variables
# #################
# CG_FOUND        | = System has NVidia Cg and it can be used.
# CG_INCLUDE_PATH | = Directory where cg.h resides
# CG_LIBRARY      | = Full path to libCg.so (Cg.DLL on WIN32)
# CG_GL_LIBRARY   | = Full path to libCgGL.so (CgGL.dll on WIN32)
# CG_COMPILER     | = Full path to cgc (cgc.exe on WIN32)
#
#
# On OSX default to using the framework version of Cg.
#####################################################################

MESSAGE("-- FINDING Cg")
IF (APPLE)
    # Finder
    INCLUDE(${CMAKE_ROOT}/Modules/CMakeFindFrameworks.cmake)
    SET(CG_FRAMEWORK_INCLUDES)
    CMAKE_FIND_FRAMEWORKS(Cg)

    IF (Cg_FRAMEWORKS)
        FOREACH(dir ${Cg_FRAMEWORKS})
            SET(CG_FRAMEWORK_INCLUDES ${CG_FRAMEWORK_INCLUDES}
                ${dir}/Headers ${dir}/PrivateHeaders)
        ENDFOREACH(dir)

        # Find the include directory
        FIND_PATH(CG_INCLUDE_PATH cg.h ${CG_FRAMEWORK_INCLUDES})

        # Add to system includes
        INCLUDE_DIRECTORIES(${CG_FRAMEWORK_INCLUDES})

        # Linking to Cg Framework
        # Linking against Cg Library
        SET(CG_LIBRARY "-framework Cg" CACHE STRING "Cg library")

        # Linking against CgGL Library
        SET(CG_GL_LIBRARY "-framework Cg" CACHE STRING "Cg GL library")
    ENDIF (Cg_FRAMEWORKS)

    # Find the Cg Compiler
    FIND_PROGRAM(CG_COMPILER cgc
                /usr/bin
                /usr/local/bin
                DOC "The Cg compiler")
ELSE (APPLE)
    # Windows
    IF (WIN32)
        FIND_PROGRAM( CG_COMPILER cgc
                    "C:/Program Files/NVIDIA Corporation/Cg/bin"
                    "C:/Program Files/Cg"
                    ${PROJECT_SOURCE_DIR}/../Cg
                    DOC "The Cg Compiler")
        IF (CG_COMPILER)
            GET_FILENAME_COMPONENT(CG_COMPILER_DIR ${CG_COMPILER} PATH)
            GET_FILENAME_COMPONENT(CG_COMPILER_SUPER_DIR ${CG_COMPILER_DIR} PATH)
        ELSE (CG_COMPILER)
            SET (CG_COMPILER_DIR .)
            SET (CG_COMPILER_SUPER_DIR ..)
        ENDIF (CG_COMPILER)

        FIND_PATH(CG_INCLUDE_PATH Cg/cg.h
                "C:/Program Files/NVIDIA Corporation/Cg/include"
                "C:/Program Files/Cg"
                ${PROJECT_SOURCE_DIR}/../Cg
                ${CG_COMPILER_SUPER_DIR}/include
                ${CG_COMPILER_DIR}
                DOC "The directory where Cg/cg.h resides")
        FIND_LIBRARY(CG_LIBRARY
                    NAMES Cg
                    PATHS
                        "C:/Program Files/NVIDIA Corporation/Cg/lib"
                        "C:/Program Files/Cg"
                        ${PROJECT_SOURCE_DIR}/../Cg
                        ${CG_COMPILER_SUPER_DIR}/lib
                        ${CG_COMPILER_DIR}
                        DOC "The Cg runtime library")
        FIND_LIBRARY( CG_GL_LIBRARY
                    NAMES CgGL
                    PATHS
                        "C:/Program Files/NVIDIA Corporation/Cg/lib"
                        "C:/Program Files/Cg"
                        ${PROJECT_SOURCE_DIR}/../Cg
                        ${CG_COMPILER_SUPER_DIR}/lib
                        ${CG_COMPILER_DIR}
                        DOC "The Cg runtime library")
    # Linux
    ELSE (WIN32)
        FIND_PROGRAM(CG_COMPILER cgc
                     /usr/bin
                     /usr/local/bin
                     DOC "The Cg Compiler")
        GET_FILENAME_COMPONENT(CG_COMPILER_DIR "${CG_COMPILER}" PATH)
        GET_FILENAME_COMPONENT(CG_COMPILER_SUPER_DIR "${CG_COMPILER_DIR}" PATH)

        FIND_PATH(CG_INCLUDE_PATH Cg/cg.h
                 /usr/include
                 /usr/local/include
                 ${CG_COMPILER_SUPER_DIR}/include
                 DOC "The directory where Cg/cg.h resides")

        FIND_LIBRARY(CG_LIBRARY Cg
                    HINTS
                        /usr/lib/x86_64-linux-gnu/
                    PATHS
                        /usr/lib64
                        /usr/lib
                        /usr/local/lib64
                        /usr/local/lib
                        ${CG_COMPILER_SUPER_DIR}/lib64
                        ${CG_COMPILER_SUPER_DIR}/lib
                        DOC "The Cg runtime library")

        FIND_LIBRARY(CG_GL_LIBRARY CgGL
                    HINTS
                        /usr/lib/x86_64-linux-gnu/
                    PATHS
                        /usr/lib64
                        /usr/lib
                        /usr/local/lib64
                        /usr/local/lib
                        ${CG_COMPILER_SUPER_DIR}/lib64
                        ${CG_COMPILER_SUPER_DIR}/lib
                        DOC "The Cg runtime library")
    ENDIF (WIN32)
ENDIF (APPLE)

IF (CG_INCLUDE_PATH)
    SET( CG_FOUND 1 CACHE STRING "Set to 1 if CG is found, 0 otherwise")
ELSE (CG_INCLUDE_PATH)
    SET( CG_FOUND 0 CACHE STRING "Set to 1 if CG is found, 0 otherwise")
ENDIF (CG_INCLUDE_PATH)

MARK_AS_ADVANCED(CG_FOUND)

# Aggregate all the libraries to CG_LIBS
SET(CG_LIBS ${CG_LIBRARY} ${CG_GL_LIBRARY})

IF(CG_FOUND)
    MESSAGE(STATUS "NVIDIA Cg Found")
    MESSAGE("-- NVIDIA Cg include directory   : ${CG_INCLUDE_PATH}")
    MESSAGE("-- NVIDIA Cg library directory   : ${CG_LIBRARY}")
    MESSAGE("-- NVIDIA CgGL library directory : ${CG_GL_LIBRARY}")
    INCLUDE_DIRECTORIES(${CG_INCLUDE_PATH})
    LINK_LIBRARIES(${CG_LIBS})
ELSE(CG_FOUND)
    MESSAGE(FATAL_ERROR "NVIDIA Cg NOT Found")
ENDIF(CG_FOUND)
