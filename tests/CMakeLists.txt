if(TL_FETCH_DEPS)
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        FIND_PACKAGE_ARGS 1.14 NAMES GTest)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
else()
    find_package(GTest 1.14 REQUIRED)
endif()
include(GoogleTest)

# tl_add_test(<name> SOURCES <files>... LABELS <labels>...)
# Labels select what each CI job runs: unit, generated, mpi, gpu, bench, abi, array_api
# (and example, for the documentation examples).
function(tl_add_test name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;LABELS")
    add_executable(${name} ${arg_SOURCES})
    target_link_libraries(${name} PRIVATE tensorlib::core GTest::gtest_main)
    tl_set_warnings(${name})
    gtest_discover_tests(${name} PROPERTIES LABELS "${arg_LABELS}")
endfunction()

add_subdirectory(core)
add_subdirectory(tools)
