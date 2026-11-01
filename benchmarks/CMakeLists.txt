if(TL_FETCH_DEPS)
    include(FetchContent)
    FetchContent_Declare(googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.tar.gz
        FIND_PACKAGE_ARGS 1.8 NAMES benchmark)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
else()
    find_package(benchmark 1.8 REQUIRED)
endif()

# Each benchmark compares library code with hand-written code; benchmarks/limits.toml sets
# how far behind the library code may be. In an optimised build, CTest checks the limits
# (label bench): the time ratios, and that the library kernels make no extra calls.
find_package(Python3 3.11 COMPONENTS Interpreter)
if(Python3_FOUND AND NOT MSVC AND CMAKE_OBJDUMP AND CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo)$")
    set(tl_check_benchmarks ON)
endif()

# tl_add_benchmark(<name> SOURCES <files>...)
function(tl_add_benchmark name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES")
    add_executable(${name} ${arg_SOURCES})
    target_link_libraries(${name} PRIVATE tensorlib::core benchmark::benchmark_main)
    tl_set_warnings(${name})
    if(tl_check_benchmarks)
        foreach(check IN ITEMS ratios codegen)
            add_test(NAME ${name}.${check}
                COMMAND Python3::Interpreter "${PROJECT_SOURCE_DIR}/tools/ci/check_benchmarks.py" ${check}
                        $<TARGET_FILE:${name}> --limits "${CMAKE_CURRENT_SOURCE_DIR}/limits.toml"
                        --compiler ${CMAKE_CXX_COMPILER_ID} --objdump "${CMAKE_OBJDUMP}")
            set_tests_properties(${name}.${check} PROPERTIES LABELS bench RUN_SERIAL ON)
        endforeach()
    endif()
endfunction()

tl_add_benchmark(bench_fixed SOURCES fixed.cpp)
