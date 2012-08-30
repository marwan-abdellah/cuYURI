# Copyright (c) 2012 Marwan Abdellah <abdellah.marwan@gmail.com>

ADD_DEFINITIONS(-D_REENTRANT)

INCLUDE_DIRECTORIES("/home/abdellah/Software/install/dcmtk")
LINK_DIRECTORIES("/home/abdellah/Software/install/lib")

find_library(FFTW_FLOAT_LIB NAMES fftw3f
  HINTS ${FFTW_ROOT}/lib
  PATHS /usr/lib /usr/local/lib /opt/local/lib
)

LINK_LIBRARIES( dcmdata
                dcmdsig
                dcmimage
                dcmimgle
                dcmjpeg
                dcmnet
                dcmpstat
                dcmqrdb
                dcmsr
                dcmtls
                dcmwlm
                ijg12
                ijg16
                ijg8
                oflog
                ofstd )
