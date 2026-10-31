# Minimum C++ standard and build system

Type: grilling
Status: resolved
Blocked by: 04, 16

## Question

Which C++ standard is the minimum, which features are allowed or banned, and which build system and package managers are supported (CMake, Spack, Conda, vcpkg)?

## Answer

Decided with the user on 2026-10-30, from [C++ standard support across target toolchains](04-toolchain-support-matrix.md).

- **Standard.** C++20 is the floor everywhere: core, backends and public headers. It matches Kokkos 5 (GCC 10.4, Clang 14 or 15 for CUDA, MSVC 19.30, NVCC 12.2, ROCm 6.2, IntelLLVM 2024.2.1). C++23 features (deducing this, `std::mdspan`, `std::expected`) appear only behind feature-test macros, as optional paths, never as requirements. The floor moves up when Kokkos moves it. There are no modules ([Techniques to cut compile times of template-heavy libraries](02-compile-time-techniques.md)).
- **Public headers.**
  - The headers a user includes may only include a whitelist of light standard headers that every toolchain at the floor provides: `<cstddef>`, `<cstdint>`, `<type_traits>`, `<concepts>`, `<utility>`, `<initializer_list>`, `<span>`, `<array>`, `<compare>`, `<limits>`, and `<cmath>` for `fixed`.
  - Banned there: `<ranges>`, `<format>`, `<print>`, `<iostream>`, `<regex>`, `<algorithm>`, `<functional>`, and any library feature missing from libstdc++ 10, because nvcc and hipcc use the system libstdc++.
  - The library's own `.cpp` files may use anything the floor supports.
  - CI checks the include list and a compile-time budget on an example TU, starting from the generator in [Prototype of compile times for lighter templates](11-compile-time-prototype.md).
- **Errors.**
  - The C++ API throws typed exceptions derived from `std::exception`: `tl::shape_error`, `tl::unsupported_operation`, `tl::device_error` and so on.
  - The C ABI for bindings catches every exception and returns a status code with a message.
  - Device kernels never throw; their failures surface at the next wait.
  - v1 does not support `-fno-exceptions`.
- **Build.**
  - CMake 3.25.2 or newer, needed for C++20 in the CUDA language, with `CMakePresets.json` per platform.
  - The install exports `find_package(tensorlib)` with targets `tensorlib::core`, `tensorlib::kokkos` and so on.
  - Dependencies (Kokkos, mdspan, Umpire, MPI) are found through `find_package`. `FetchContent` is opt-in for developers (`TL_FETCH_DEPS=ON`), and nothing is vendored by default. No second build system.
- **Packages.** Official recipes, maintained and tested in CI:
  - Spack, with variants `+cuda`, `+rocm`, `+sycl`, `+mpi` and `+umpire`;
  - conda-forge, with CPU and CUDA builds.

  vcpkg can come later.

## Amendment (2026-10-30)

[Zero-copy interop with PETSc and Tpetra](17-petsc-tpetra-interop.md): the optional components `tensorlib::petsc` and `tensorlib::tpetra` add Spack variants `+petsc` and `+trilinos`, and tensorlib, PETSc built with Kokkos, and Trilinos must resolve to one Kokkos installation in both Spack and conda-forge.

## Amendment (2026-10-31)

[Coding style and extension points](29-coding-style-and-extension-points.md): public headers are strict C++20 with no C++23 conditional paths. C++23 enters the API all at once, replacing C++20 code, when the floor moves. The library's own `.cpp` files may raise their standard separately, without keeping two versions of the code.

## Amendment (2026-10-31, floor)

Decided with the user while implementing [Build skeleton and repository layout](../../core-implementation/issues/01-build-skeleton.md). `tl::site` captures `std::source_location` in public headers ([Error reporting and debug mode](24-error-reporting-and-debug-mode.md)), and `<source_location>` is missing from libstdc++ 10 (checked with GCC 10.5). The host floor rises above Kokkos 5's to:

- GCC 11.1 (Kokkos warns on 11.0);
- Clang 16, with libstdc++ 11 or newer or with libc++ 16 or newer. libstdc++ defines `std::source_location` only when the compiler has `__builtin_source_location` (Clang 15), libc++ ships the header from version 16 (libc++ 15 checked), and Clang 15 evaluates `std::source_location::current()` used as a default argument at the callee: the CI test of [Hosted CI matrix and header checks](../../core-implementation/issues/02-ci-matrix-and-header-checks.md) got the line of the default argument instead of the caller's on Clang 15.0.7, and the right line on Clang 16.0.6;
- Apple Clang 15;
- MSVC 19.30, unchanged.

`<source_location>` joins the public header whitelist. A machine whose system libstdc++ is version 10 loads a newer GCC (`gcc-toolset`, modules) for nvcc and hipcc. NVCC 12.2.140 with GCC 11.4 as host compiler compiles `<tl/tl.hpp>` next to a kernel and captures the caller's line with a defaulted `std::source_location` (checked in the `nvidia/cuda:12.2.2-devel-ubuntu22.04` image while implementing [Hosted CI matrix and header checks](../../core-implementation/issues/02-ci-matrix-and-header-checks.md)). hipcc is not checked yet. Capturing with `__builtin_FILE()`, `__builtin_LINE()` and `__builtin_FUNCTION()` stays the fallback if a target toolchain cannot follow.
