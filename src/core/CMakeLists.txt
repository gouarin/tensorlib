configure_file("${PROJECT_SOURCE_DIR}/include/tl/version.hpp.in"
    "${PROJECT_BINARY_DIR}/include/tl/version.hpp" @ONLY)

add_library(tensorlib_core version.cpp)
add_library(tensorlib::core ALIAS tensorlib_core)
set_target_properties(tensorlib_core PROPERTIES
    EXPORT_NAME core
    OUTPUT_NAME tensorlib
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    # The library exports functions only, so exporting every symbol is enough for a Windows DLL.
    WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_compile_features(tensorlib_core PUBLIC cxx_std_20)
target_include_directories(tensorlib_core PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
if(TL_ENABLE_ASSERTIONS)
    target_compile_definitions(tensorlib_core PUBLIC TL_ENABLE_ASSERTIONS)
endif()
tl_set_warnings(tensorlib_core)

install(TARGETS tensorlib_core EXPORT tensorlibTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/tl"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.hpp")
install(FILES "${PROJECT_BINARY_DIR}/include/tl/version.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/tl")
