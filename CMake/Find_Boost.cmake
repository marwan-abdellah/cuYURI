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

MESSAGE("-- FINDING Boost")

# Find Boost package
FIND_PACKAGE(Boost REQUIRED)

IF(Boost_FOUND)
    MESSAGE(STATUS "FOUND, Boost")
    MESSAGE("-- Boost include directory : ${BOOST_INCLUDE_DIR}")
    MESSAGE("-- Boost library directory : ${BOOST_LIB_DIR}")
ELSE(Boost_FOUND)
    MESSAGE(FATAL_ERROR "Boost NOT Found")
ENDIF(Boost_FOUND)

# Set Boost heuristic directories
SET(BOOST_INC_DIR ${Boost_INCLUDE_DIR})
SET(BOOST_LIB_DIR ${Boost_LIBRARY_DIRS})

# Timer library
FIND_LIBRARY(BOOST_TIMER_LIB NAMES boost_timer-mt
    HINTS ${BOOST_LIB_DIR}
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/local/lib
    )

# File system library
FIND_LIBRARY(BOOST_FILESYSTEM_LIB NAMES boost_filesystem-mt
    HINTS ${BOOST_LIB_DIR}
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/local/lib
    )

# System library
FIND_LIBRARY(BOOST_SYSTEM_LIB NAMES boost_system-mt
    HINTS ${BOOST_LIB_DIR}
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/local/lib
    )

# Thread library
FIND_LIBRARY(BOOST_THREAD_LIB NAMES boost_thread-mt
    HINTS ${BOOST_LIB_DIR}
    PATHS
        /usr/lib
        /usr/local/lib
        /opt/local/lib
    )

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(BOOST DEFAULT_MSG
                BOOST_TIMER_LIB
                BOOST_FILESYSTEM_LIB
                BOOST_SYSTEM_LIB
                BOOST_THREAD_LIB
                BOOST_INC_DIR
)

# Include directories
INCLUDE_DIRECTORIES(${BOOST_INC_DIR})

# Aggregate all the libraries in BOOST_LIBS
SET(BOOST_LIBS  ${BOOST_TIMER_LIB}
                ${BOOST_FILESYSTEM_LIB}
                ${BOOST_SYSTEM_LIB}
                ${BOOST_THREAD_LIB})

# Link against the boost libraries
LINK_LIBRARIES(${BOOST_LIBS})
