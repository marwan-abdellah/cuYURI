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

MESSAGE("-- FINDING ImageMagick")

FIND_PACKAGE(ImageMagick REQUIRED)

IF(ImageMagick_FOUND)
    MESSAGE(STATUS "ImageMagick Found")
    MESSAGE("-- ImageMagick include directory : ${ImageMagick_INCLUDE_DIRS}")
    MESSAGE("-- ImageMagick library directory : ${ImageMagick_LIBRARIES}")
ELSE(ImageMagick_FOUND)
    MESSAGE(FATAL_ERROR "ImageMagick NOT Found")
ENDIF(ImageMagick_FOUND)

SET(ImageMagick_INC_DIR ${ImageMagick_INCLUDE_DIRS})
SET(ImageMagick_LIB_DIR ${ImageMagick_LIBRARIES})



# Find Magick++ library
FIND_LIBRARY(Magick_LIB NAMES Magick++
    PATHS
        /opt/local/lib
        /usr/lib
        /usr/local/lib
    )

# Find MagickCore library
FIND_LIBRARY(MagickCore_LIB NAMES MagickCore
    PATHS
        /opt/local/lib
        /usr/lib
        /usr/local/lib
    )

# Find MagickWand library
FIND_LIBRARY(MagickWand_LIB NAMES MagickWand
    PATHS
        /opt/local/lib
        /usr/lib
        /usr/local/lib
    )

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(ImageMagick DEFAULT_MSG
                                  Magick_LIB
                                  MagickCore_LIB
                                  MagickWand_LIB)
# Library aggregation
SET(ImageMagick_LIBS ${Magick_LIB} ${MagickCore_LIB} ${MagickWand_LIB})

# Add to the system tree
INCLUDE_DIRECTORIES(${ImageMagick_INC_DIR})
INCLUDE_DIRECTORIES("/opt/local/include/ImageMagick")

# Link against the ImageMagick libraries
LINK_LIBRARIES(${ImageMagick_LIBS})

MESSAGE("-- ImageMagick libraries : ${ImageMagick_LIBS}")
