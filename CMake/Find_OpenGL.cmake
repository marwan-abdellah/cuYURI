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

MESSAGE("-- FINDING OpenGL")

IF(APPLE)
    # Base directory
    INCLUDE_DIRECTORIES ("/System/Library/Frameworks")

    # Find the OpenGL library
    FIND_LIBRARY(OPENGL_LIBRARY OpenGL REQUIRED)

    # Advanced variable
    MARK_AS_ADVANCED(OPENGL_LIBRARY)

    # Adding the include directory to the systems
    SET(OPENGL_INCLUDE_DIR "/usr/X11R6/include/")
    INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})

    # Adding the library directories
    SET(OPENGL_LIB_DIR "/usr/X11R6/lib")
    LINK_DIRECTORIES(${OPENGL_LIB_DIR})

    # Checking if OPENGL was found or not
    IF(NOT OPENGL_LIBRARY STREQUAL "")
        MESSAGE(STATUS "OpenGL Found")
        MESSAGE("-- OpenGL include directory : ${OPENGL_INCLUDE_DIR}")
        MESSAGE("-- OpenGL library directory : ${OPENGL_LIBRARY}")
    ELSE(NOT OPENGL_LIBRARY STREQUAL "")
        MESSAGE(FATAL_ERROR "-- OpenGL NOT Found")
    ENDIF(NOT OPENGL_LIBRARY STREQUAL "")

    # Linking against OpenGL
    LINK_LIBRARIES(${OPENGL_LIBRARY})
ELSE(APPLE)
    # Linux
    FIND_PACKAGE(OpenGL REQUIRED)
    IF(OPENGL_FOUND)
        MESSAGE(STATUS "OpenGL Found")
        MESSAGE("-- OpenGL include directory : ${OPENGL_INCLUDE_DIR}")
        MESSAGE("-- OpenGL library directory : ${OPENGL_LIBRARY}")
        INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR}/GL)
        LINK_LIBRARIES(${OPENGL_LIBRARY})
    ELSE(OPENGL_FOUND)
        MESSAGE(FATAL_ERROR "-- OpenGL NOT Found")
    ENDIF(OPENGL_FOUND)
ENDIF(APPLE)





