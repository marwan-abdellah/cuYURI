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

MESSAGE("-- FINDING GLEW")

IF(APPLE)
    # Base directory
    INCLUDE_DIRECTORIES ("/System/Library/Frameworks")

    # Find the GLEW library
    FIND_LIBRARY(GLEW_LIBRARY GLEW REQUIRED)

    # Advanced variable
    MARK_AS_ADVANCED(GLEW_LIBRARY)

    # Adding the include directory to the systems
    SET(GLEW_INCLUDE_DIR "/usr/X11R6/include/")
    INCLUDE_DIRECTORIES(${GLEW_INCLUDE_DIR})

    # Adding the library directories
    SET(GLEW_LIB_DIR "/usr/X11R6/lib")
    LINK_DIRECTORIES(${GLEW_LIB_DIR})

    # Checking if GLEW was found or not
    IF(NOT GLEW_LIBRARY STREQUAL "")
        MESSAGE(STATUS "GLEW Found")
        MESSAGE("-- GLEW include directory : ${GLEW_INCLUDE_DIR}")
        MESSAGE("-- GLEW library directory : ${GLEW_LIBRARY}")
    ELSE(NOT GLEW_LIBRARY STREQUAL "")
        MESSAGE(FATAL_ERROR "GLEW NOT Found")
    ENDIF(NOT GLEW_LIBRARY STREQUAL "")

    # Linking against GLEW
    LINK_LIBRARIES(${GLEW_LIBRARY})
ELSE(APPLE)
    # Linux
    FIND_PACKAGE(GLEW REQUIRED)
    IF(GLEW_FOUND)
        MESSAGE(STATUS "GLEW Found")
        MESSAGE("-- GLEW include directory : ${GLEW_INCLUDE_DIR}")
        MESSAGE("-- GLEW library directory : ${GLEW_LIBRARY}")
        INCLUDE_DIRECTORIES(${GLEW_INCLUDE_DIR}/GL)
        LINK_LIBRARIES(${GLEW_LIBRARY})
    ELSE(GLEW_FOUND)
        MESSAGE(FATAL_ERROR "GLEW NOT Found")
    ENDIF(GLEW_FOUND)
ENDIF(APPLE)




