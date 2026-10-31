# C++ standard support across target toolchains

Research for ticket `issues/04-toolchain-support-matrix.md`.

Read in October 2026. cppreference, clang.llvm.org, learn.microsoft.com, kokkos.org, docs.nvidia.com, rocm.docs.amd.com, intel.com and developer.apple.com were blocked by the sandbox firewall. Where possible the same data is cited from the source files the published pages are built from (GitHub repos of LLVM, MicrosoftDocs, microsoft/STL, Kokkos, CMake). gcc.gnu.org was reachable. NVIDIA facts come from web search snippets of docs.nvidia.com and from NVIDIA's own CCCL repo; they are marked as such.

## 1. Short answer

- Kokkos 4.x requires C++17 (C++20, 23 and 26 accepted). Kokkos 5.x requires C++20 (23 and 26 accepted). Kokkos 5 minimums: GCC 10.4, Clang 14 (15 for CUDA), IntelLLVM 2022.0 (2024.2.1 for SYCL), NVCC 12.2, ROCm/HIPCC 6.2, NVHPC 22.3, MSVC 19.30 (VS 2022 17.0), CMake 3.22 (3.25.2 for C++20 in the CUDA language).
- C++20 core language features (concepts, `consteval`, `constexpr` allocation, class-type NTTPs) work on every compiler that Kokkos 5 accepts, with small gaps at the oldest Clang versions.
- C++23 is not a usable baseline today. MSVC still has no stable `/std:c++23` switch, only `/std:c++23preview` and `/std:c++latest`. nvcc accepts `-std=c++23` only since CUDA 13.3. `std::mdspan` arrived in libstdc++ only with GCC 16 (2026). libc++ still lacks `views::chunk`, `views::cartesian_product`, `<stacktrace>`, and full `constexpr <cmath>`.
- Modules are not usable in a library that must build with nvcc: the CUDA docs say modules are not supported in CUDA C++, in host or device code. CMake also does not scan modules for AppleClang.
- For a multidimensional view type, use `Kokkos::mdspan` (Kokkos bundles the reference implementation in namespace `Kokkos`), not `std::mdspan`. It works on every Kokkos backend, including in device code.

## 2. Kokkos requirements

### C++ standard

| Kokkos | Required standard | Accepted | Source |
|---|---|---|---|
| 4.7.04 | C++17 (default when `CMAKE_CXX_STANDARD` is unset) | 17, 20, 23, 26 | [kokkos_test_cxx_std.cmake L76-L93](https://github.com/kokkos/kokkos/blob/4.7.04/cmake/kokkos_test_cxx_std.cmake#L76-L93), [kokkos_compiler_id.cmake L157-L160](https://github.com/kokkos/kokkos/blob/4.7.04/cmake/kokkos_compiler_id.cmake#L157-L160) |
| 5.2.2 | C++20 (default) | 20, 23, 26 | [kokkos_pick_cxx_std.cmake L7](https://github.com/kokkos/kokkos/blob/5.2.2/cmake/kokkos_pick_cxx_std.cmake#L7), [kokkos_test_cxx_std.cmake L76-L89](https://github.com/kokkos/kokkos/blob/5.2.2/cmake/kokkos_test_cxx_std.cmake#L76-L89) |

Kokkos 5.0.0 changelog: "Require C++20 (#8372). Implies a new set of minimum compiler versions." ([CHANGELOG.md, 5.0.0](https://github.com/kokkos/kokkos/blob/5.2.2/CHANGELOG.md#500)). Kokkos 5.2.0 adds "Enable C++23 support with NVCC (available since CUDA 13.3.0)" ([PR #9264](https://github.com/kokkos/kokkos/pull/9264)). The PR notes that under C++23, math functions on types convertible to `float` return `std::float32_t`, and that nvcc warns more about `std::pair` in device code.

### Minimum compilers (enforced at configure time)

Values from `cmake/kokkos_compiler_id.cmake`, which fails the configure step below these versions. They match the [requirements page source](https://github.com/kokkos/kokkos-core-wiki/blob/main/docs/source/get-started/requirements.rst).

| Compiler | Kokkos 4.7 with C++17 | Kokkos 4.7 with C++20 | Kokkos 5.2 |
|---|---|---|---|
| GCC | 8.2.0 | 10.4.0 | 10.4.0 (warns on 11.0, use 11.1+) |
| Clang (CPU) | 8.0.0 | 14.0.0 | 14.0.0 |
| Clang (CUDA) | 10.0.0 | 14.0.0 | 15.0.0 |
| IntelLLVM (CPU) | 2021.1.1 | 2022.0.0 | 2022.0.0 |
| IntelLLVM (SYCL) | 2024.2.1 | 2024.2.1 | 2024.2.1 |
| NVCC | 11.0.0 | 12.0.0 | 12.2.0 |
| HIPCC (ROCm) | 5.2.0 | 5.2.0 | 6.2.0 |
| NVHPC | 22.3 | 22.3 | 22.3 |
| MSVC | 19.29 | 19.30 | 19.30 |
| Intel Classic, IBM XL | rejected | rejected | rejected |

Sources: [4.7.04 kokkos_compiler_id.cmake L157-L182](https://github.com/kokkos/kokkos/blob/4.7.04/cmake/kokkos_compiler_id.cmake#L157-L182), [5.2.2 kokkos_compiler_id.cmake L155-L235](https://github.com/kokkos/kokkos/blob/5.2.2/cmake/kokkos_compiler_id.cmake#L155-L235).

AppleClang has no minimum check in 5.2.2. The 4.x docs list AppleClang 8.0. Kokkos 5.2.2 CI builds with AppleClang on `macos-14` and `macos-15` runners, C++20, Serial and Threads backends only ([continuous-integration-osx.yml](https://github.com/kokkos/kokkos/blob/5.2.2/.github/workflows/continuous-integration-osx.yml)). Windows CI covers VS 2022 with CUDA 12.4.1 and `windows-2025` runners ([continuous-integration-windows.yml](https://github.com/kokkos/kokkos/blob/5.2.2/.github/workflows/continuous-integration-windows.yml)).

### CMake

- Kokkos 4.x: CMake 3.16 ([CMakeLists.txt L1](https://github.com/kokkos/kokkos/blob/4.7.04/CMakeLists.txt#L1)); 3.21.1 for NVC++ (requirements page).
- Kokkos 5.x: CMake 3.22 ([CMakeLists.txt L1](https://github.com/kokkos/kokkos/blob/5.2.2/CMakeLists.txt#L1)); 3.25.2 for C++20 in the CUDA language (changelog, [#8402](https://github.com/kokkos/kokkos/pull/8402)); 3.28.2 for the experimental modules option ([kokkos_enable_options.cmake L118-L125](https://github.com/kokkos/kokkos/blob/5.2.2/cmake/kokkos_enable_options.cmake#L118-L125)).

### Kokkos features related to this question

- `Kokkos_ENABLE_EXPERIMENTAL_CXX20_MODULES` (default OFF) exports a `kokkos.core` module. 5.0 lists "Complete C++20 module support". 5.2 says the module "can be used with latest MSVC and gcc-16" ([#9148](https://github.com/kokkos/kokkos/pull/9148)).
- Kokkos 5.0 re-enabled its `mdspan`-based `View`. Kokkos ships the reference `mdspan` in `tpls/mdspan` with `MDSPAN_IMPL_STANDARD_NAMESPACE` set to `Kokkos` and the proposed namespace `Kokkos::Experimental` ([tpls/mdspan/include/mdspan/mdspan.hpp L19-L25](https://github.com/kokkos/kokkos/blob/5.2.2/tpls/mdspan/include/mdspan/mdspan.hpp#L19-L25)). `View` converts to and from `std::mdspan` when the standard library has it (changelog, [#6830](https://github.com/kokkos/kokkos/pull/6830)).
- Kokkos 5.1 exports its type traits as C++20 concepts ([#8494](https://github.com/kokkos/kokkos/pull/8494)).
- `Kokkos_ENABLE_CUDA_CONSTEXPR` (nvcc `--expt-relaxed-constexpr`) is OFF by default with nvcc and ON with Clang-CUDA ([kokkos_enable_options.cmake L168-L175](https://github.com/kokkos/kokkos/blob/5.2.2/cmake/kokkos_enable_options.cmake#L168-L175)). Without it, nvcc does not let device code call host `constexpr` functions, such as most of the host standard library.

## 3. What the GPU compilers are underneath

The device compilers inherit their C++ front end from a host compiler or an LLVM release, and on Linux they use the system libstdc++. So library feature support depends on which GCC is installed.

| Toolchain | Front end | Standard library | Source |
|---|---|---|---|
| hipcc, ROCm 6.2 / 6.3 | LLVM 18 | host libstdc++ on Linux | [ROCm/llvm-project rocm-6.2.0 LLVMVersion.cmake](https://github.com/ROCm/llvm-project/blob/rocm-6.2.0/cmake/Modules/LLVMVersion.cmake) |
| hipcc, ROCm 6.4 | LLVM 19 | same | [rocm-6.4.0](https://github.com/ROCm/llvm-project/blob/rocm-6.4.0/cmake/Modules/LLVMVersion.cmake) |
| hipcc, ROCm 7.0 / 7.1 | LLVM 20 | same | [rocm-7.0.0](https://github.com/ROCm/llvm-project/blob/rocm-7.0.0/cmake/Modules/LLVMVersion.cmake) |
| hipcc, ROCm 7.2.x | LLVM 22 | same | [rocm-7.2.0](https://github.com/ROCm/llvm-project/blob/rocm-7.2.0/cmake/Modules/LLVMVersion.cmake) |
| icpx, oneAPI 2025.0 | LLVM 19 | host libstdc++ on Linux, MSVC STL on Windows | [intel/llvm v6.0.0 release notes](https://github.com/intel/llvm/releases/tag/v6.0.0) ("oneAPI DPC++/C++ Compiler version 2025.0 leverages codebase from sycl-rel-6_0_0") and [LLVMVersion.cmake at v6.0.0](https://github.com/intel/llvm/blob/v6.0.0/cmake/Modules/LLVMVersion.cmake) |
| icpx, oneAPI 2025.1 | LLVM 20 | same | [v6.1.0](https://github.com/intel/llvm/releases/tag/v6.1.0) |
| icpx, oneAPI 2025.2 / 2025.3 | LLVM 21 | same | [v6.2.0](https://github.com/intel/llvm/releases/tag/v6.2.0), [v6.3.0](https://github.com/intel/llvm/releases/tag/v6.3.0) |
| nvcc 12.x / 13.x | EDG front end; host code compiled by GCC, Clang, MSVC or nvc++ | host library; NVIDIA supports only libstdc++ on Linux | CUDA Linux installation guide, via search snippet |

The intel/llvm open-source tags are "the closest" to the named oneAPI release, not identical to it. The LLVM major of the product compiler is therefore approximate.

nvcc details (from search snippets of NVIDIA docs, not read directly):

- The legacy CUDA C++ Programming Guide says all C++20 features are supported in nvcc 12.0 and later, with listed exceptions: modules are not supported in host or device code, and coroutines are not supported in device code. `-std=c++20` needs GCC 10+, Clang 10+, Visual Studio 2022+ or nvc++ 20.7+ as host compiler ([CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/), [C++ Language Support appendix](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html)).
- CUDA 13.3 adds official C++23 support to nvcc via `--std=c++23` ([NVIDIA blog on CUDA 13.3](https://developer.nvidia.com/blog/nvidia-cuda-13-3-enhances-gpu-development-with-tile-programming-in-c-compiler-autotuning-and-python-updates/); also confirmed by Kokkos [#9264](https://github.com/kokkos/kokkos/pull/9264)). I could not read the per-feature C++23 table.
- NVIDIA's CCCL CI tests these host compilers: CUDA 12.x with GCC 7 to 14, Clang 14 to 19, MSVC 2019 and 2022; CUDA 13.0 with GCC 11 to 15, Clang 15 to 20, MSVC 2019 and 2022 ([cccl ci/matrix.yaml L97-L112](https://github.com/NVIDIA/cccl/blob/main/ci/matrix.yaml)). CCCL 3.x supports C++17 and C++20 dialects ([cccl README, C++ Dialects](https://github.com/NVIDIA/cccl/blob/main/README.md#c-dialects)). This is a proxy for the official host compiler table, which I could not read.

## 4. Feature by toolchain matrix

Version numbers are the first release with the feature. "lib" rows depend on the standard library, not the compiler: for nvcc, hipcc and icpx on Linux, read the libstdc++ column for the GCC installed on the machine.

MSVC versions map to `_MSC_VER` as follows: VS 2019 16.3 = 19.23, 16.10 = 19.29, VS 2022 17.0 = 19.30, 17.3 = 19.33, 17.4 = 19.34, 17.5 = 19.35, 17.7 = 19.37, 17.9 = 19.39, 17.12 = 19.42, 17.13 = 19.43, 17.14 = 19.44, VS 2026 18.0 = 19.50 (MSVC Build Tools 14.50, per the conformance page).

| Feature | GCC / libstdc++ | Clang / libc++ | MSVC / MS STL | nvcc | hipcc (ROCm) | icpx |
|---|---|---|---|---|---|---|
| Concepts, language (P0734) | 10 | 10; last piece (P0848, conditionally trivial special members) in 19 | 16.3 | 12.0+ with `-std=c++20` | yes from ROCm 6.2 (LLVM 18); complete from 6.4 (LLVM 19) | yes; complete from 2025.0 (LLVM 19) |
| `<concepts>` library (P0898) | 10.1 | 13 | 16.3 | host lib | host lib | host lib |
| C++20 `<ranges>` (P0896) | 10.1 (later DRs P2325 in 11.3, P2415 in 12.1) | 15 | 16.10 (first only under `/std:c++latest`, footnote "20abi") | host lib; host-only code | host lib | host lib |
| C++23 ranges: `ranges::to` | 14.1 | 17 | 17.4 | host lib | host lib | host lib |
| C++23 ranges: `zip`, `enumerate` | 13.1 | `zip` 22, `enumerate` 23 | `zip` complete 17.7, `enumerate` 17.7 | host lib | host lib | host lib |
| C++23 ranges: `chunk`/`slide`, `cartesian_product` | 13.1 | not implemented | `chunk` 17.3, `cartesian_product` 17.7 | host lib | host lib | host lib |
| Modules, language (P1103) | 11 with `-fmodules` (partial); CMake scanning from GCC 14 | 15 (some parts in 17); CMake scanning from Clang 16 | 16.8; CMake scanning from 14.34 (VS 17.4) | not supported (host or device) | not verified; treat as unusable | not verified |
| `import std` (P2465) | 15.1; CMake: GCC 15 (16 on macOS) | 19; CMake: Clang 18.1.2+ | 17.5; CMake: 14.36 (VS 17.6), Ninja only | no (modules unsupported) | not verified | not verified |
| `std::mdspan` (P0009) | 16.1 | 18 | 17.9 | host lib only, not `__device__`; use `Kokkos::mdspan` | host lib (GCC 16 libstdc++ needed); use `Kokkos::mdspan` | same |
| `std::expected` (P0323) | 12.1 (monadic ops 13.1) | 16 (monadic 17) | 17.3 | host lib, host code | host lib | host lib |
| Deducing this (P0847) | 14 | 18 | 17.13 (`/std:c++latest` or `/std:c++23preview`) | 13.3+ with `-std=c++23` | ROCm 6.2+ (LLVM 18) | 2025.0+ |
| Multidimensional `operator[]` (P2128) | 12 | 15 | 17.12 (C++23 mode) | 13.3+ | ROCm 6.2+ | yes |
| `if consteval` (P1938) | 12 | 14 | 17.14 | 13.3+ | yes | yes |
| `consteval` (P1073) | 10 partial, 11 full | 17 (listed as complete) | 16.10 | 12.0+ | ROCm 6.2+ | 2025.0+ |
| `constexpr` new/delete (P0784) | 10 | 10 | 16.9 | 12.0+ | yes | yes |
| `constexpr std::vector` / `std::string` | 12.1 | 15 | 16.10 | host lib | host lib | host lib |
| C++23: non-literal variables in `constexpr` (P2242) | 12 | 15 | 17.14 | 13.3+ | yes | yes |
| C++23: `static constexpr` in `constexpr` functions (P2647) | 13 | 16 | not listed | 13.3+ | yes | yes |
| C++23: relaxing `constexpr` restrictions (P2448) | 13 | 19 | no | 13.3+ | ROCm 6.4+ | 2025.0+ |
| C++23: `constexpr std::unique_ptr` (P2273) | 12.1 | 16 | 17.3 | host lib | host lib | host lib |
| C++23: `constexpr <cmath>` (P0533) | partial (no version in table) | in progress | 14.52 (Preview) | host lib | host lib | host lib |
| `std::format` (P0645) | 13.1 | 14 | 16.10 | host lib | host lib | host lib |
| `std::print` (P2093) | 14.1 | 18 | 17.7 | host lib | host lib | host lib |
| `std::flat_map` (P0429) | 15.1 | 20 | 14.51 (VS 2026 18.6) | host lib | host lib | host lib |
| `<stacktrace>` (P0881) | 12.1 | not implemented | 17.4 | host lib | host lib | host lib |

For hipcc and icpx, the language rows follow the LLVM release in section 3. AMD and Intel may patch their forks; I did not check their release notes.

Sources for the matrix:

- GCC language: [gcc.gnu.org/projects/cxx-status.html](https://gcc.gnu.org/projects/cxx-status.html)
- libstdc++: [libstdc++ status](https://gcc.gnu.org/onlinedocs/libstdc++/manual/status.html)
- Clang language: [clang/www/cxx_status.html](https://github.com/llvm/llvm-project/blob/main/clang/www/cxx_status.html) (source of clang.llvm.org/cxx_status.html)
- libc++: [Cxx20Papers.csv](https://github.com/llvm/llvm-project/blob/main/libcxx/docs/Status/Cxx20Papers.csv), [Cxx23Papers.csv](https://github.com/llvm/llvm-project/blob/main/libcxx/docs/Status/Cxx23Papers.csv)
- MSVC language and library: [visual-cpp-language-conformance.md](https://github.com/MicrosoftDocs/cpp-docs/blob/main/docs/overview/visual-cpp-language-conformance.md) (source of the learn.microsoft.com page, dated 2026-09-01). The C++23 library part of that table lacks some rows, so MS STL dates for `mdspan`, `print`, `zip`, `enumerate`, `cartesian_product`, `import std` and `flat_map` come from the [STL changelog](https://github.com/microsoft/STL/wiki/Changelog) and [VS 2022 changelog](https://github.com/microsoft/STL/wiki/VS-2022-Changelog).
- MSVC switches: [std-specify-language-standard-version.md](https://github.com/MicrosoftDocs/cpp-docs/blob/main/docs/build/reference/std-specify-language-standard-version.md). `/std:c++23preview` exists since VS 2022 17.13 Preview 4, and "will be removed when the `/std:c++23` switch is implemented". The STL changelog says C++23 features "generally require the `/std:c++23` or `/std:c++latest` compiler options".
- CMake modules: [cmake-cxxmodules.7.rst](https://github.com/Kitware/CMake/blob/master/Help/manual/cmake-cxxmodules.7.rst), "Compiler Support" and "`import std` Support". AppleClang is not in the list. Ubuntu before 26.04 ships broken `libstdc++.modules.json`.

## 5. Apple Clang

I could not read Xcode release notes (developer.apple.com was blocked), so I have no primary source for Apple Clang versions. What I can state:

- Apple Clang tracks upstream Clang and libc++ with a lag and its own version numbers. A community summary ([luncliff/vcpkg-registry#624](https://github.com/luncliff/vcpkg-registry/issues/624), secondary) groups Xcode 16.0-16.2 as "AppleClang 16" and Xcode 16.3+ and 26.x as "AppleClang 17", and notes that Apple does not publish the upstream LLVM version per point release.
- Xcode 26.4 ships libc++ additions such as `aligned_accessor` for `mdspan` (P2897R7), per a search snippet of the [Xcode 26.4 release notes](https://developer.apple.com/documentation/xcode-release-notes/xcode-26_4-release-notes). This means `std::mdspan` is present in current Xcode.
- CMake does not support module scanning with AppleClang (section 4).
- Kokkos tests AppleClang from `macos-14` (C++20, Serial and Threads).

Treat Apple Clang as "about one upstream Clang release behind" and check each feature with a feature-test macro (`__cpp_explicit_this_parameter`, `__cpp_lib_mdspan`, `__cpp_lib_expected`, ...) in CI.

## 6. Versions on HPC systems

I did not read facility pages for this. The points below are framing for the decision, not verified facts.

- On Linux, nvcc, hipcc and icpx all use the system libstdc++ (NVIDIA states it supports only libstdc++). The library feature set is therefore set by the GCC module loaded on the machine, not by the GPU toolkit version.
- Enterprise base OS compilers lag: RHEL 8 ships GCC 8 and RHEL 9 ships GCC 11 as system compilers (from memory, not verified); newer GCC comes from toolsets or modules.
- From the toolkit side, the oldest versions Kokkos 5 still accepts are CUDA 12.2, ROCm 6.2 (LLVM 18) and oneAPI 2022.0 (CPU) or 2024.2.1 (SYCL).

## 7. What this means for tensorlib

- Basing on Kokkos 5 fixes C++20 as the floor. Concepts, `consteval`, `constexpr` allocation, class-type NTTPs and C++20 `<ranges>` (host code) are usable on all targets. The oldest accepted Clang (14) lacks complete `consteval` (17) and P0848 (19); that is a minor risk.
- Not safe at the floor: `std::format` (libstdc++ 13), `std::expected` (libstdc++ 12), C++23 ranges, `std::print`, `std::flat_map`, `<stacktrace>`.
- C++23 as a required standard would exclude: MSVC with a stable switch (none yet), nvcc before CUDA 13.3, libstdc++ before 16 for `std::mdspan`, and libc++ for several ranges adaptors.
- Deducing this needs GCC 14, Clang 18, MSVC 17.13 in preview mode and nvcc 13.3 in C++23 mode. It cannot be relied on in the core today; it could sit behind `__cpp_explicit_this_parameter`.
- Modules cannot be the primary interface because nvcc does not support them. An optional module wrapper, like Kokkos' experimental `kokkos.core`, is possible for host-only users later.
- For the tensor view type, build on `Kokkos::mdspan` / `Kokkos::extents` / layout types. They match the standard API, run in device code, and convert to `std::mdspan` where it exists.
- For error types in host code, a small in-house `expected`-like type, or a feature-gated alias to `std::expected`, avoids raising the libstdc++ floor to 12.
- Device code cannot call host standard library functions under nvcc unless `--expt-relaxed-constexpr` is on, and Kokkos leaves it off by default with nvcc. Kernels should use Kokkos facilities (`Kokkos::pair`, `Kokkos::Array`, Kokkos math) rather than `std::` ones.

## 8. Gaps

- Exact host compiler ranges for CUDA 13.x and the per-feature C++23 table for nvcc (docs.nvidia.com was blocked).
- AMD and Intel release notes for their forks; module support in hipcc and icpx.
- Apple Clang per-version feature data.
- Compiler versions installed at specific HPC centers.
