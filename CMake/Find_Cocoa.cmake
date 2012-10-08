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


MESSAGE("-- FINDING Cocoa")

IF(APPLE)
    # Base directory
    INCLUDE_DIRECTORIES ("/System/Library/Frameworks")

    # Find the Cocoa library
    FIND_LIBRARY(COCOA_LIBRARY Cocoa REQUIRED)

    # Advanced variable
    MARK_AS_ADVANCED(COCOA_LIBRARY)

    # Adding the include directory to the systems
    SET(COCOA_INCLUDE_DIR "/usr/X11R6/include/")
    INCLUDE_DIRECTORIES(${COCOA_INCLUDE_DIR})

    # Adding the library directories
    SET(COCOA_LIB_DIR "/usr/X11R6/lib")
    LINK_DIRECTORIES(${COCOA_LIB_DIR})

    # Checking if Cocoa was found or not
    IF(NOT COCOA_LIBRARY STREQUAL "")
        MESSAGE(STATUS "Cocoa Found")
        MESSAGE("-- Cocoa include directory : ${COCOA_INCLUDE_DIR}")
        MESSAGE("-- Cocoa library directory : ${COCOA_LIBRARY}")
    ELSE(NOT COCOA_LIBRARY STREQUAL "")
        MESSAGE(FATAL_ERROR "-- Cocoa NOT Found")
    ENDIF(NOT COCOA_LIBRARY STREQUAL "")

    # Linking against Cocoa
    LINK_LIBRARIES(${COCOA_LIBRARY})
ENDIF(APPLE)




