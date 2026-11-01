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

# tl_add_benchmark(<name> SOURCES <files>...)
function(tl_add_benchmark name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES")
    add_executable(${name} ${arg_SOURCES})
    target_link_libraries(${name} PRIVATE tensorlib::core benchmark::benchmark_main)
    tl_set_warnings(${name})
endfunction()

tl_add_benchmark(bench_fixed SOURCES fixed.cpp)
