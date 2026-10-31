# PROTOTYPE - throwaway build for the "Lowering of short segments" ticket.
cmake_minimum_required(VERSION 3.22)
project(tl_proto_30 C CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(LLVM REQUIRED CONFIG)
# Ubuntu's libkokkos-dev ships a broken KokkosTargets.cmake (see prototype 27).
find_package(OpenMP REQUIRED)
add_library(kokkos_sys INTERFACE)
target_include_directories(kokkos_sys INTERFACE /usr/include/kokkos)
target_link_libraries(kokkos_sys INTERFACE kokkoscontainers kokkoscore OpenMP::OpenMP_CXX dl)
find_package(benchmark REQUIRED)

add_library(proto_jit STATIC jit.cpp)
target_include_directories(proto_jit SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(proto_jit PUBLIC ${LLVM_DEFINITIONS})
target_link_libraries(proto_jit PUBLIC LLVM)

foreach(exe check bench)
  add_executable(${exe} ${exe}.cpp)
  target_compile_options(${exe} PRIVATE -O3 -mcpu=apple-m3)
  target_link_libraries(${exe} PRIVATE proto_jit kokkos_sys)
endforeach()
target_link_libraries(bench PRIVATE benchmark::benchmark)
