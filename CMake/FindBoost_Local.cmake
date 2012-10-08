# Copyright (c) 2012 Marwan Abdellah <abdellah.marwan@gmail.com>

FIND_PACKAGE(Boost REQUIRED)

if(Boost_FOUND)
  MESSAGE(STATUS FOUND, Boost)
  MESSAGE(BOOST_INCLUDE_DIR->${Boost_INCLUDE_DIR})
  MESSAGE(BOOST_LIB_DIR->${Boost_LIBRARY_DIRS})
endif()

# Set Boost heuristic directories 
set(BOOST_INC_DIR ${Boost_INCLUDE_DIR})
set(BOOST_LIB_DIR ${Boost_LIBRARY_DIRS})

find_library(BOOST_TIMER_LIB NAMES boost_timer-mt
  HINTS ${BOOST_LIB_DIR}
  PATHS /usr/lib 
	/usr/local/lib 
	/opt/local/lib
)

find_library(BOOST_CHRONO_MT_LIB NAMES boost_chrono-mt
  HINTS ${BOOST_LIB_DIR}
  PATHS /usr/lib 
	/usr/local/lib 
	/opt/local/lib
)

find_library(BOOST_CHRONO_LIB NAMES boost_chrono
  HINTS ${BOOST_LIB_DIR}
  PATHS /usr/lib 
	/usr/local/lib 
	/opt/local/lib
)

find_library(BOOST_FILESYSTEM_MT_LIB NAMES boost_filesystem-mt
  HINTS ${BOOST_LIB_DIR}
  PATHS /usr/lib 
	/usr/local/lib 
	/opt/local/lib
)

find_library(BOOST_FILESYSTEM_LIB NAMES boost_filesystem
  HINTS ${BOOST_LIB_DIR}
  PATHS /usr/lib 
	/usr/local/lib 
	/opt/local/lib
)

find_library(BOOST_SYSTEM_MT_LIB NAMES boost_system-mt
  HINTS ${BOOST_LIB_DIR}
  PATHS /usr/lib 
	/usr/local/lib 
	/opt/local/lib
)

find_library(BOOST_SYSTEM_LIB NAMES boost_system
  HINTS ${BOOST_LIB_DIR}
  PATHS /usr/lib 
	/usr/local/lib 
	/opt/local/lib
)

find_library(BOOST_THREAD_MT_LIB NAMES boost_thread-mt
  HINTS ${BOOST_LIB_DIR}
  PATHS /usr/lib 
	/usr/local/lib 
	/opt/local/lib
)

find_library(BOOST_THREAD_LIB NAMES boost_thread
  HINTS ${BOOST_LIB_DIR}
  PATHS /usr/lib 
	/usr/local/lib 
	/opt/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BOOST DEFAULT_MSG 
		BOOST_TIMER_LIB
		BOOST_INC_DIR
)

# Include directories 
INCLUDE_DIRECTORIES(${BOOST_INC_DIR})   

# Link Boost timer libraries to the application 
LINK_LIBRARIES(${BOOST_TIMER_LIB})
LINK_LIBRARIES(${BOOST_CHRONO_MT_LIB})
LINK_LIBRARIES(${BOOST_FILESYSTEM_MT_LIB})
LINK_LIBRARIES(${BOOST_SYSTEM_MT_LIB})
LINK_LIBRARIES(${BOOST_THREAD_MT_LIB})


if(BOOST_FOUND)
  message(STATUS "Found Boost in 
	${Boost_INCLUDE_DIRS} 
        ${BOOST_TIMER_LIB}")

ENDIF(BOOST_FOUND)


