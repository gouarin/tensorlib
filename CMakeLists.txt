cmake_minimum_required(VERSION 3.25.2)

project(tensorlib
    VERSION 0.1.0
    DESCRIPTION "Lazy, fused and distributed tensors for HPC"
    LANGUAGES CXX)

option(TL_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
option(TL_BUILD_BENCHMARKS "Build the benchmarks" OFF)
option(TL_FETCH_DEPS "Fetch missing dependencies with FetchContent (for developers)" OFF)
option(TL_WARNINGS_AS_ERRORS "Turn compiler warnings into errors (CI only)" OFF)
option(TL_ENABLE_ASSERTIONS "Host-side assertions in public headers; does not change the ABI" OFF)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(tl_warnings)
include(GNUInstallDirs)

add_subdirectory(src/core)

if(TL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(examples)
endif()

if(TL_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

include(CMakePackageConfigHelpers)
set(TL_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/tensorlib")
configure_package_config_file(cmake/tensorlibConfig.cmake.in
    "${PROJECT_BINARY_DIR}/tensorlibConfig.cmake"
    INSTALL_DESTINATION "${TL_INSTALL_CMAKEDIR}")
# Before 1.0, a minor release may break the API.
write_basic_package_version_file("${PROJECT_BINARY_DIR}/tensorlibConfigVersion.cmake"
    COMPATIBILITY SameMinorVersion)
install(FILES
    "${PROJECT_BINARY_DIR}/tensorlibConfig.cmake"
    "${PROJECT_BINARY_DIR}/tensorlibConfigVersion.cmake"
    DESTINATION "${TL_INSTALL_CMAKEDIR}")
install(EXPORT tensorlibTargets
    NAMESPACE tensorlib::
    DESTINATION "${TL_INSTALL_CMAKEDIR}")
