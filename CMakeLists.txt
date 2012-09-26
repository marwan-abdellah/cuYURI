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

# Minimum required CMake version 
cmake_minimum_required(VERSION 2.6)

# Fourier Volume Rendeing
PROJECT(cuYuri)

# Add CMake modules 
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)

# FFTW
SET(FFTW_ROOT ${PROJECT_SOURCE_DIR}/../install) 
FIND_PACKAGE(FFTW REQUIRED)
IF(FFTW_FOUND)
  MESSAGE(STATUS "FFTW found...")
  MESSAGE("-- FFTW include directory : ${FFTW_INCLUDE_DIR}")
  MESSAGE("-- FFTW library directory : ${FFTW_LIBRARIES}")
  INCLUDE_DIRECTORIES(${FFTW_INCLUDE_DIR})
  LINK_LIBRARIES(${FFTW_LIBRARIES})
ELSE(FFTW_FOUND)
  MESSAGE(FATAL_ERROR "FFTW not found")
ENDIF(FFTW_FOUND)

# OpenGL
FIND_PACKAGE(OpenGL REQUIRED)

IF(OPENGL_FOUND)
    MESSAGE(STATUS "OpenGL found...")
    MESSAGE("-- OpenGL include directory : ${OPENGL_INCLUDE_DIR}")
    MESSAGE("-- OpenGL library directory : ${OPENGL_LIBRARY}")
    INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})
    LINK_LIBRARIES(${OPENGL_LIBRARY})
ELSE(OPENGL_FOUND)
    MESSAGE(FATAL_ERROR "OpenGL not found")
ENDIF(OPENGL_FOUND)

# GLUT
FIND_PACKAGE(GLUT REQUIRED)
IF(GLUT_FOUND)
    MESSAGE(STATUS  "GLUT found...")
    MESSAGE("GLUT include directory : ${GLUT_INCLUDE_DIR}")
    MESSAGE("GLUT library directory : ${GLUT_LIBRARY}")
    INCLUDE_DIRECTORIES(${GLUT_INCLUDE_DIR}/GL)
    LINK_LIBRARIES(${GLUT_LIBRARY})
ELSE(GLUT_FOUND)
    MESSAGE(FATAL_ERROR "GLUT not found")
ENDIF(GLUT_FOUND)

# GLEW
INCLUDE(FindGLEW)
LINK_LIBRARIES(${GLEW_LIB})
INCLUDE_DIRECTORIES(${GLEW_INCLUDE_DIR})

# Add the heade files to the include directories
INCLUDE_DIRECTORIES("${OPENGL_INCLUDE_DIR}")

# Includes
INCLUDE(Includes)

SET(CPP_SOURCES
Logging/Logging.cpp
Memory/Memory_1D.cpp
Memory/Memory_2D.cpp
Memory/Memory_3D.cpp
Timers/Boost.cpp
Timers/CUDA.cpp
Utilities/XL.cpp
Utilities/Utils.cpp
CUDA/cuMemory_1D_Array.cpp
CUDA/cuDataTransfer_1D_Array.cpp
CUDA/cuKernelConf.cpp
Magick/Image.cpp
OpenGL/Glut.cpp
OpenGL/OpenGL.cpp
OpenGL/GLEW.cpp)


LIST(APPEND CPP_SOURCES
Dependencies/Excel/BasicExcel.cpp
Dependencies/Excel/BasicExcel.hpp
Dependencies/Excel/ExcelFormat.cpp
)

# Include directories 
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# CUDA kernels directory 
SET(CUDA_KERNELS_DIR ${CMAKE_SOURCE_DIR}/CUDA
                     ${CMAKE_SOURCE_DIR}/CUDA/FFT
                     ${CMAKE_SOURCE_DIR}/CUDA/Interfaces

                        )
INCLUDE_DIRECTORIES(${CUDA_KERNELS_DIR})


#ADD_SUBDIRECTORY(doc)
INCLUDE(${CMAKE_SOURCE_DIR}/doc)


#########################################################
#! Compiler Flags
#########################################################
set(CMAKE_C_FLAGS " -fpermissive -fPIC -O3 -pthread -pipe -DLINUX -D_BOOL -DREQUIRE_IOSTREAM -Wno-deprecated -fno-gnu-keywords -w")

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")

INCLUDE(${CMAKE_SOURCE_DIR}/CUDA/cuKernels.cmake)

CUDA_ADD_LIBRARY(cuYURI ${CUDA_SOURCES} ${CPP_SOURCES})
SET_PROPERTY(TARGET cuYURI PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIRECTORY}/libcuYURI.a)




#########################################################
# Compile Examples
#########################################################
INCLUDE(${CMAKE_SOURCE_DIR}/Examples/FillArrayRND/CMakeLists.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/Examples/CreateXLDocument/CMakeLists.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/Examples/Add_1D_Arrays/CMakeLists.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/Examples/DisplayArrayGL/CMakeLists.cmake)


# --------------------------------------------------------
# Test Projects
# --------------------------------------------------------
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/CUDA/MatMul/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/CUDA/FFT_Shift/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/LibXL/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/Array/cufftComplex/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/Array/fftwComplex/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/Array/Real/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/Timers/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/Templates/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Tests/CUDA/MaxSubArray/CMakeLists.cmake)

# --------------------------------------------------------
# Benchmarking Projects
# --------------------------------------------------------
#INCLUDE(${CMAKE_SOURCE_DIR}/Benchmarking/FFT_Shift/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Benchmarking/cuFFTShift_2D/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Benchmarking/cuFFTShift_3D/CMakeLists.cmake)
#INCLUDE(${CMAKE_SOURCE_DIR}/Benchmarking/FFT_Shift/2D/Real/CMakeLists.cmake)

# FVR
#INCLUDE(${CMAKE_SOURCE_DIR}/FourierVolumeRenderer/CMakeLists.cmake)
