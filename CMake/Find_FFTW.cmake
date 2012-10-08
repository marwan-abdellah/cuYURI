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

MESSAGE("-- FINDING FFTW")

MARK_AS_ADVANCED(FFTW_ROOT)

# Find the FFT package
FIND_PACKAGE(FFTW REQUIRED)

IF(FFTW_FOUND)
    MESSAGE(STATUS "FFTW Found")
    MESSAGE("-- FFTW include directory : ${FFTW_INCLUDE_DIR}")
    MESSAGE("-- FFTW library directory : ${FFTW_LIBRARIES}")
    INCLUDE_DIRECTORIES(${FFTW_INCLUDE_DIR})
    LINK_LIBRARIES(${FFTW_LIBRARIES})
ELSE(FFTW_FOUND)
    # Find the header file
    FIND_PATH(FFTW_INCLUDE_DIR "fftw3.h"
        HINTS ${FFTW_ROOT}/include
        /usr/include
        /usr/local/include
        /opt/local/include
        )

    # Find the single precision library "f=float"
    FIND_LIBRARY(FFTW_LIB_SINGLE NAMES fftw3f
        HINTS ${FFTW_ROOT}/lib
        PATHS
            /usr/lib
            /usr/local/lib
            /opt/local/lib
        )

    # Find the double precision library
    FIND_LIBRARY(FFTW_LIB_DOUBLE NAMES fftw3
        HINTS ${FFTW_ROOT}/lib
        PATHS
            /usr/lib
            /usr/local/lib
            /opt/local/lib
        )

    # Group the libraries to link against the single and double precision at once
    SET(FFTW_LIBS ${FFTW_LIB_SINGLE} ${FFTW_LIB_DOUBLE})

    INCLUDE(FindPackageHandleStandardArgs)
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(FFTW DEFAULT_MSG
                                      FFTW_LIB_SINGLE
                                      FFTW_LIB_DOUBLE
                                      FFTW_INCLUDE_DIR
                                      )

    IF(FFTW_FOUND)
        MESSAGE(STATUS "FFTW Found")
        MESSAGE("-- FFTW include directory : ${FFTW_INCLUDE_DIR}")
        MESSAGE("-- FFTW library directory : ${FFTW_LIBS}")
        INCLUDE_DIRECTORIES(${FFTW_INCLUDE_DIR})
        LINK_LIBRARIES(${FFTW_LIBS})
    ELSE(FFTW_FOUND)
        MESSAGE(FATAL_ERROR "FFTW NOT Found")
    ENDIF(HWLOC_FOUND)
ENDIF(FFTW_FOUND)
