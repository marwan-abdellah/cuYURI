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

MESSAGE("-- FINDING GLUT")

IF(APPLE)
    # Base directory
    INCLUDE_DIRECTORIES ("/System/Library/Frameworks")

    # Find the GLUT library
    FIND_LIBRARY(GLUT_LIBRARY GLUT REQUIRED)

    # Advanced variable
    MARK_AS_ADVANCED(GLUT_LIBRARY)

    # Adding the include directory to the systems
    SET(GLUT_INCLUDE_DIR "/usr/X11R6/include/")
    INCLUDE_DIRECTORIES(${GLUT_INCLUDE_DIR})

    # Adding the library directories
    SET(GLUT_LIB_DIR "/usr/X11R6/lib")
    LINK_DIRECTORIES(${GLUT_LIB_DIR})

    # Checking if GLUT was found or not
    IF(NOT GLUT_LIBRARY STREQUAL "")
        MESSAGE(STATUS "GLUT Found")
        MESSAGE("-- GLUT include directory : ${GLUT_INCLUDE_DIR}")
        MESSAGE("-- GLUT library directory : ${GLUT_LIBRARY}")
    ELSE(NOT GLUT_LIBRARY STREQUAL "")
        MESSAGE(FATAL_ERROR "GLUT NOT Found")
    ENDIF(NOT GLUT_LIBRARY STREQUAL "")

    # Linking against GLUT
    LINK_LIBRARIES(${GLUT_LIBRARY})
ELSE(APPLE)
    # Linux
    FIND_PACKAGE(GLUT REQUIRED)
    IF(GLUT_FOUND)
        MESSAGE(STATUS "GLUT Found")
        MESSAGE("-- GLUT include directory : ${GLUT_INCLUDE_DIR}")
        MESSAGE("-- GLUT library directory: ${GLUT_LIBRARY}")
        INCLUDE_DIRECTORIES(${GLUT_INCLUDE_DIR}/GL)
        LINK_LIBRARIES(${GLUT_LIBRARY})
    ELSE(GLUT_FOUND)
        MESSAGE(FATAL_ERROR "GLUT NOT Found")
    ENDIF(GLUT_FOUND)
ENDIF(APPLE)




