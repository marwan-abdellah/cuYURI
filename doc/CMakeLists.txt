FIND_PACKAGE(Doxygen)
OPTION(BUILD_DOCS "Build DOXYGEN Documentation" ON)
MARK_AS_ADVANCED(BUILD_DOCS)

IF(BUILD_DOCS AND DOXYGEN_FOUND)
  # Update the config file with the correct version number
  CONFIGURE_FILE(documentation-config.doxygen.in
                 ${CMAKE_CURRENT_BINARY_DIR}/documentation-config.doxygen
                 @ONLY)
  ADD_CUSTOM_TARGET(doxygen ALL
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/documentation-config.doxygen
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation using doxygen for cuYURI" VERBATIM)

  MAKE_DIRECTORY(${CMAKE_CURRENT_BINARY_DIR}/html) 
  INSTALL(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html
          DESTINATION share/cuYURI-0.1 COMPONENT docs)
ENDIF(BUILD_DOCS AND DOXYGEN_FOUND)
