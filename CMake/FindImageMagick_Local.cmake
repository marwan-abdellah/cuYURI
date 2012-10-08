# Copyright (c) 2012 Marwan Abdellah <abdellah.marwan@gmail.com>

FIND_PACKAGE(ImageMagick REQUIRED)

if(ImageMagick_FOUND)
  MESSAGE(STATUS FOUND, ImageMagick)
endif()

# Set Boost heuristic directories 
set(ImageMagick_INC_DIR "/opt/local/include/ImageMagick")
set(ImageMagick_LIB_DIR "/usr/lib")

find_library(Magick_LIB NAMES Magick++
  PATHS /opt/local/lib
        /usr/lib
	/usr/local/lib 

)

find_library(MagickCore_LIB NAMES MagickCore
  PATHS /opt/local/lib
        /usr/lib
        /usr/local/lib

)

find_library(MagickWand_LIB NAMES MagickWand
  PATHS /opt/local/lib
        /usr/lib
        /usr/local/lib

)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ImageMagick DEFAULT_MSG Magick_LIB MagickCore_LIB MagickWand_LIB)

# Include directories 
MESSAGE($ImageMagick_INC_DIR->${ImageMagick_INC_DIR})
INCLUDE_DIRECTORIES(${ImageMagick_INC_DIR})   
INCLUDE_DIRECTORIES("${ImageMagick_INC_DIR}/Magick++")
INCLUDE_DIRECTORIES("/opt/local/include/ImageMagick")


# Link Boost timer libraries to the application 
LINK_LIBRARIES(${Magick_LIB} ${MagickCore_LIB} ${MagickWand_LIB})
