# PROTOTYPE - throwaway build for the "Prototype of multi-output fusion" ticket.
cmake_minimum_required(VERSION 3.22)
project(tl_proto_09 CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Kokkos REQUIRED)
find_package(OpenMP REQUIRED)
find_package(benchmark REQUIRED)

add_executable(check check.cpp)
target_link_libraries(check PRIVATE Kokkos::kokkos)

add_executable(bench bench.cpp)
target_include_directories(bench PRIVATE ${XTENSOR_INCLUDE} ${XTL_INCLUDE} ${XSIMD_INCLUDE})
target_compile_definitions(bench PRIVATE XTENSOR_USE_XSIMD XTENSOR_USE_OPENMP)
target_link_libraries(bench PRIVATE Kokkos::kokkos OpenMP::OpenMP_CXX benchmark::benchmark)
