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

#########################################################
#! Minimum required CMake version
#########################################################
cmake_minimum_required(VERSION 2.8)

#########################################################
#! Project name
#########################################################
PROJECT(cuYURI)

#########################################################
#! Add CMake customized modules
#########################################################
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/CMake)

#########################################################
#! Find packages
#########################################################
# @FFTW
INCLUDE(Find_FFTW)

# @Cocoa for Apple
INCLUDE(Find_Cocoa)

# @OpenGL
INCLUDE(Find_OpenGL)

# @GLUT
INCLUDE(Find_GLUT)

# @GLEW
INCLUDE(Find_GLEW)

# @X11 supporting libraries
INCLUDE(Find_X11)

# @Cg NVIDIA
INCLUDE(Find_Cg)

# @CUDA
INCLUDE(Find_CUDA)

# @CUDA GPU Computing SDK
INCLUDE(Find_CUDA_SDK)

# @OpenMP
INCLUDE(Find_OpenMP)

# @ImageMagick
INCLUDE(Find_ImageMagick)

# @Boost
INCLUDE(Find_Boost)



SET(CPP_SOURCES
Logging/Logging.cpp
Memory/Memory_1D.cpp
Memory/Memory_2D.cpp
Memory/Memory_3D.cpp
Timers/Boost.cpp
Timers/CUDA.cpp
#Utilities/XL.cpp
Utilities/Utils.cpp
CUDA/cuMemory_1D_Array.cpp
CUDA/cuDataTransfer_1D_Array.cpp
CUDA/cuKernelConf.cpp
Magick/Image.cpp
OpenGL/OpenGL.cpp
OpenGL/GLEW.cpp
OpenGL/GLUT.cpp
Volume/Loader.cpp
Volume/Volume.cpp
Volume/TestVolume.cpp
OpenGL/Cg.cpp
OpenGL/RayCaster.cpp
OpenGL/RayCaster_GL.cpp
OpenGL/VolumeRayCaster.cpp
OpenGL/SimpleVolumeSlicer.cpp
)

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
#INCLUDE(${CMAKE_SOURCE_DIR}/Examples/CreateXLDocument/CMakeLists.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/Examples/Add_1D_Arrays/CMakeLists.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/Examples/DisplayArrayGL/CMakeLists.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/Examples/VolumeSlicer/CMakeLists.cmake)
INCLUDE(${CMAKE_SOURCE_DIR}/Examples/VolumeRayCaster/CMakeLists.cmake)

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




#########################################################
#! Generating executable
#########################################################
CUDA_ADD_EXECUTABLE(run ${CMAKE_SOURCE_DIR}/Visualization/VolumeSlicer_3D/VolumeRenderPyramid.cpp)

#########################################################
#! Link the executable against the cuYURI library
#########################################################
TARGET_LINK_LIBRARIES(run cuYURI)
