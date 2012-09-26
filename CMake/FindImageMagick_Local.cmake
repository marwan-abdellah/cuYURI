# Copyright (c) 2012 Marwan Abdellah <abdellah.marwan@gmail.com>

FIND_PACKAGE(ImageMagick REQUIRED)

if(ImageMagick_FOUND)
  MESSAGE(STATUS FOUND, ImageMagick)
endif()

# Set Boost heuristic directories 
set(ImageMagick_INC_DIR "/usr/include/ImageMagick")
set(ImageMagick_LIB_DIR "/usr/lib")

find_library(Magick_LIB NAMES Magick++
  PATHS /usr/lib 
	/usr/local/lib 
	/opt/local/lib
)

find_library(MagickCore_LIB NAMES MagickCore
  PATHS /usr/lib
        /usr/local/lib
        /opt/local/lib
)

find_library(MagickWand_LIB NAMES MagickWand
  PATHS /usr/lib
        /usr/local/lib
        /opt/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ImageMagick DEFAULT_MSG Magick_LIB MagickCore_LIB MagickWand_LIB)

# Include directories 
INCLUDE_DIRECTORIES(${ImageMagick_INC_DIR})   

# Link Boost timer libraries to the application 
LINK_LIBRARIES(${Magick_LIB} ${MagickCore_LIB} ${MagickWand_LIB})
