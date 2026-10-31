# C++ standard support across target toolchains

Type: research
Status: resolved

## Question

Which C++20 and C++23 features are usable on GCC, Clang, Apple Clang, MSVC, nvcc, hipcc and icpx in versions found on current HPC systems, and what C++ standard and compilers do Kokkos 4.x and 5.x require? Produce a feature by toolchain matrix (concepts, ranges, modules, `std::mdspan`, `std::expected`, deducing this, `constexpr` extensions).

## Answer

- Kokkos 4.x requires C++17. Kokkos 5.x requires C++20 and enforces GCC 10.4, Clang 14 (15 for CUDA), MSVC 19.30, NVCC 12.2, ROCm 6.2, IntelLLVM 2022.0 (2024.2.1 for SYCL) and CMake 3.22 (3.25.2 for C++20 in the CUDA language).
- C++20 core language (concepts, `consteval`, `constexpr` allocation) works on every toolchain Kokkos 5 accepts. Library features such as `std::format` and `std::expected` need a newer libstdc++ than the GCC 10.4 floor, and nvcc, hipcc and icpx use the system libstdc++.
- C++23 is not a viable baseline: MSVC has no stable `/std:c++23` yet, nvcc accepts C++23 only from CUDA 13.3, `std::mdspan` needs GCC 16, and libc++ lacks some ranges adaptors. Deducing this needs GCC 14, Clang 18, MSVC 17.13 (preview mode) and CUDA 13.3.
- Modules are out: nvcc does not support them, and CMake does not scan them for AppleClang. Use `Kokkos::mdspan` rather than `std::mdspan`; it runs in device code on every backend.

Findings: [research/04-toolchain-support-matrix.md](../research/04-toolchain-support-matrix.md)
