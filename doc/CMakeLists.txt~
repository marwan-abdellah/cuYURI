# @author  Stefan Eilemann
# @remarks Copyright (c) BBP/EPFL 2011; All rights reserved. Do not distribute without further notice.

find_package(Doxygen)
option(BUILD_DOCS "Build doxygen documentation" ON)
mark_as_advanced(BUILD_DOCS)

if(BUILD_DOCS AND DOXYGEN_FOUND)
  # Update the config file with the correct version number
  configure_file(documentation-config.doxygen.in
                 ${CMAKE_CURRENT_BINARY_DIR}/documentation-config.doxygen
                 @ONLY)
  add_custom_target(doxygen ALL
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/documentation-config.doxygen
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation using doxygen" VERBATIM)

  make_directory(${CMAKE_CURRENT_BINARY_DIR}/html) # needed for install:
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html
          DESTINATION share/cuYuri-0.1 COMPONENT docs)
endif(BUILD_DOCS AND DOXYGEN_FOUND)
