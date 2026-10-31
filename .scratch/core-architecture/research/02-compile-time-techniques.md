# Techniques to cut compile times of template-heavy libraries

Research for ticket `issues/02-compile-time-techniques.md`.

Read in October 2026. The sandbox firewall blocked clang.llvm.org, libcxx.llvm.org, cmake.org, learn.microsoft.com, docs.nvidia.com, intel.com, rocm.docs.amd.com, developer.apple.com, cppreference, gitlab.com (Eigen), bitbucket.org (Blaze) and most blogs. Where possible, facts are cited from the source files those pages are built from (GitHub repos of LLVM, CMake, MicrosoftDocs, microsoft/STL, Kokkos, and the authors' own blog repos). gcc.gnu.org was reachable. Facts that come only from web search snippets are marked "(snippet)". Toolchain versions overlap with `research/04-toolchain-support-matrix.md`.

## 1. Short answer

- Modules cannot be the compile-time strategy of the core. nvcc does not support them in host or device code. No GPU offload compiler (nvcc, hipcc, icpx/SYCL) supports them, and Apple Clang has neither `import std` nor CMake module scanning. GCC 16 still calls its modules support experimental. The core must compile as plain headers. A module wrapper (`import tensorlib`) can be an optional extra for host-only builds, the way Kokkos 5 does it.
- Modules do not remove the main cost of an expression-template library anyway. A module caches parsing and the instantiations done inside the module, but every user TU still instantiates and optimizes its own expression types. Published gains are 10 to 50% on real projects, and some cases get slower.
- `extern template` and explicit instantiation (ETI) help only for a closed, nameable set of types: leaf containers and named kernels over a fixed list of value types, ranks and memory spaces. They cannot reach expression trees or lambdas. Kokkos core removed its ETI in 2020 because looking up the pre-instantiated specialization cost as much as compiling the kernel. Kokkos Kernels keeps ETI for its BLAS-like kernels, where the type list is closed.
- Concepts versus SFINAE is a readability and diagnostics choice, not a compile-time one. The only published micro-benchmark shows differences that depend on the compiler and on the exact SFINAE form, sometimes favouring `requires` and sometimes not. Its author says real code should not choose based on it.
- The strongest levers are structural: keep heavy standard headers (`<iostream>`, `<format>`/`<print>`, `<ranges>`, `<meta>`) out of core headers, keep instantiation depth low, and move the variable part of the work behind a type-erased boundary. The run-time graph plus JIT family is the extreme case of that last point. PCH beats `import std` on GCC 16 for small TUs.
- Measure in CI with `-ftime-trace` (Clang, also for HIP), `-ftime-report` (GCC), `nvcc --fdevice-time-trace` with NVIDIA's Compile Time Advisor, and MSVC Build Insights.

## 2. Modules and `import std`, per toolchain

| Toolchain | Named modules | `import std` | Notes |
|---|---|---|---|
| GCC | Experimental, needs `-fmodules` (GCC 16) | GCC 15+ (std and std.compat, also in C++20 mode) | GCC 16 adds `--compile-std-module` and translates `#include` of importable std headers into `import` |
| Clang + libc++ | Usable; CMake scans since Clang 16 | Clang 18.1.2+ with installed modules; libc++ docs still say "experimental" | Reduced BMI is the default since Clang 22 |
| Apple Clang (Xcode) | No CMake scanning for `AppleClang` | Not supported | Homebrew/upstream LLVM on macOS works |
| MSVC | Usable since VS 17.4 (toolset 14.34) for CMake | VS 17.5+ (CMake: toolset 14.36+), Ninja generator only | Mixing `#include <vector>` and `import std` in one file is not supported |
| nvcc | Not supported, host or device | Not supported | Stated in the CUDA programming guide and CUDA 12 release notes |
| hipcc / amdclang++ | No AMD documentation of support | No | Same offload problem as Clang CUDA (see below) |
| icpx (oneAPI) | No release-note entry; CMake does not scan for `IntelLLVM` | No | CMake 4.5 adds scanning and `import std` for the open-source DPC++ (`DPClang`) 7.0+, host side |

Sources and details:

- GCC 15: "C++ Modules have been greatly improved", and the std library gains "`std` and `std.compat` modules (also supported for C++20)" ([GCC 15 changes](https://gcc.gnu.org/gcc-15/changes.html)).
- GCC 16: "C++20 modules support is still experimental and must be enabled by -fmodules". New option `--compile-std-module` "builds the `<bits/stdc++.h>` header unit and the `std` and `std.compat` modules before compiling any source files". "GCC now transparently translates an `#include` of any importable standard library header into an `import`". GCC 16 also switches the default to `-std=gnu++20` ([GCC 16 changes](https://gcc.gnu.org/gcc-16/changes.html)).
- libc++: "Modules are an experimental feature. [...] not considered stable nor complete". Using installed modules "requires Clang 18.1.2 or later". Known Clang issue: including headers after `import std` may fail ([libcxx/docs/Modules.md](https://github.com/llvm/llvm-project/blob/main/libcxx/docs/Modules.md), published at libcxx.llvm.org/Modules.html).
- Clang: "As of Clang 22.x, the Reduced BMI is enabled by default". Known issue #61465: `import` followed by `#include` of the same declarations is rejected ([clang/docs/StandardCPlusPlusModules.md](https://github.com/llvm/llvm-project/blob/main/clang/docs/StandardCPlusPlusModules.md)).
- CMake (master, shipping as 4.5, at rc2 on 2026-10-30): scanning for MSVC 14.34+, Clang 16+, clang-cl 19.1+, GCC 14+. `import std` for Clang 18.1.2+ (libc++ or libstdc++), MSVC 14.36+, GCC 15+ (GCC 16+ on macOS). "Only the Ninja Generators currently support `import std`". Ubuntu releases before 26.04 ship a broken `libstdc++.modules.json`. Scanned sources are excluded from unity builds ([cmake-cxxmodules.7.rst](https://github.com/Kitware/CMake/blob/master/Help/manual/cmake-cxxmodules.7.rst)). CMake 4.5 release notes: "`import std` is now available for all C++ 20 and later code with `CXX_SCAN_FOR_MODULES` enabled", so the experimental gate goes away in 4.5 ([Help/release/4.5.rst](https://github.com/Kitware/CMake/blob/master/Help/release/4.5.rst)). The compiler files show that `AppleClang-CXX.cmake` and `IntelLLVM-CXX.cmake` define no scan-deps rule, while `DPClang` reuses the Clang one ([Modules/Compiler](https://github.com/Kitware/CMake/tree/master/Modules/Compiler)).
- MSVC: the `import std` tutorial requires "Visual Studio 2022 17.5 or later" and says importing the whole library "is faster than bringing in a single header file". It also says not to mix `#include <vector>` with `import std` ([tutorial-import-stl-named-module.md](https://github.com/MicrosoftDocs/cpp-docs/blob/main/docs/cpp/tutorial-import-stl-named-module.md)). The MSVC STL 14.53 preview adds Clang support for the `std` modules ([STL Changelog](https://github.com/microsoft/STL/wiki/Changelog)).
- nvcc: the CUDA guide says modules are not supported in CUDA C++, in host or device code. A forum post quoting the CUDA 12 release notes says "Modules are not supported for both host and device code" (snippet: [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/), [NVIDIA forum](https://forums.developer.nvidia.com/t/will-c-20-modules-support-arrive/279895)). A Kokkos maintainer, Sep 2025: "So far, I'm not aware of any GPU backend supporting C++20 modules" ([kokkos#8474](https://github.com/kokkos/kokkos/issues/8474)).
- Why offload is hard: Clang issue [#57862](https://github.com/llvm/llvm-project/issues/57862) "Modules won't work for the NVPTX target" is still open. Artem Belevich (Clang CUDA): "CUDA compilation is actually N compilations, each with different set of preprocessor macros set, different target CPU [...] I do not think a single precompiled header set/module would be able to handle that". Module initializers also have no place in the GPU execution model. HIP and SYCL use the same split host/device compilation, so the same argument applies (inference: no AMD or Intel document confirms or denies it).
- Apple Clang: the mp-units support table lists "C++ modules: None" and "`import std;`: None" for apple-clang, and the same "None" for gcc and MSVC modules, because mp-units cannot build its own modules there. It also reports that Apple Clang 17 (Xcode 16.3 to 26.2-beta) cannot build mp-units at all ([mp-units cpp_compiler_support.md](https://github.com/mpusz/mp-units/blob/master/docs/getting_started/cpp_compiler_support.md)).
- icpx: Intel's 2026 release notes (2026.0 to 2026.2, Oct 2026) mention nothing about modules (snippet: [oneAPI DPC++/C++ release notes 2026](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2026.html)).

### What Kokkos does

Kokkos 5.0 lists "Complete C++20 module support", behind the option `Kokkos_ENABLE_EXPERIMENTAL_CXX20_MODULES` (default OFF, needs CMake 3.28.2). It exports `kokkos.core` from `Kokkos_Core.cppm`. Kokkos 5.2 fixes it for "latest MSVC and gcc-16". The Graph module is skipped on Clang below 20 because Clang 19 "terminates with many parsing issues" ([CHANGELOG.md](https://github.com/kokkos/kokkos/blob/develop/CHANGELOG.md), [kokkos_enable_options.cmake](https://github.com/kokkos/kokkos/blob/develop/cmake/kokkos_enable_options.cmake), [core/src/CMakeLists.txt](https://github.com/kokkos/kokkos/blob/develop/core/src/CMakeLists.txt)). In the prototype issue, the module file listing exported symbols reached "around 560 lines". The maintainers noted that macros not in `Kokkos_Macros.hpp` are invisible to module users, and that specializations of `Kokkos` templates from other modules become a problem ([kokkos#8081](https://github.com/kokkos/kokkos/issues/8081)). These are the same problems tensorlib would hit with user specializations of its traits.

## 3. Published measurements

### Modules

- Clang docs explain why the gain is smaller with optimization on. At `-O0`, imported code only goes through semantic analysis. With optimization, imported function bodies are still processed "again and again in importee units by optimizations (including IPO itself)", so "the build speedup at higher optimization levels may be lower than expected" ([StandardCPlusPlusModules.md, "How modules speed up compilation"](https://github.com/llvm/llvm-project/blob/main/clang/docs/StandardCPlusPlusModules.md)).
- Chuanqi Xu (Clang modules maintainer, Aug 2025): "The data I have obtained from practice ranges from 25% to 45%", and "most reports [...] are between 10% and 50%". Slowdowns come from lower build parallelism and duplicate declarations across module units. A toy case where `std::vector<std::string>` is instantiated inside the module: 0.516 s to 0.077 s (7.7x), with instantiation time "more than 20 times lower". Clang does not cache `constexpr`/`consteval` evaluation across modules (issue 61425) ([post source](https://github.com/ChuanqiXu9/ChuanqiXu9.github.io/blob/HEAD/docs/_posts/2025-08-14-C++20-Modules.en.md)).
- Chuanqi Xu (Sep 2026), Seastar built with an implicit `import std` (Clang, `-j8`, median of 3): wall time 92.2 s to 81.1 s (preload) and 71.6 s (module map), so 12% and 22% faster ([post source](https://github.com/ChuanqiXu9/ChuanqiXu9.github.io/blob/HEAD/docs/_posts/2026-09-01-Implicit-Std-Modules.en.md)).
- Rubén Pérez (Boost), clang-19, release build, Asio server, module build time included: 1 TU 9.1 s (modules) vs 6.9 s (headers), 7 TUs 21.0 s vs 24.7 s. Modules lose below about 4 TUs. His conclusion: "Instantiation time isn't affected by modules" ([modules.adoc](https://github.com/anarthal/cppblog/blob/HEAD/posts/modules.adoc)). Boost.Mp11 test suite (about 200 TUs): 2 min 10 s with headers, 39 s with modules ([modules3.adoc](https://github.com/anarthal/cppblog/blob/HEAD/posts/modules3.adoc)). Boost parked the effort until CMake `import std` is stable and MSVC bugs are fixed.
- Vittorio Romeo, GCC 16.0.1 release build, C++26 reflection examples: `import std` 279.5 ms vs PCH 91.7 ms on a small TU, 605.7 ms vs 628.0 ms on a large one. "PCH drastically outperforms modules for smaller headers like `<meta>`, while pretty much ties modules with large includes such as `<print>`" ([post source](https://github.com/vittorioromeo/vittorioromeo.com/blob/HEAD/content/_pages/index/_entries/180_blog_2016_md/39_refl_compiletime.md), published at vittorioromeo.com).
- MSVC 2019, per-header include cost: the standard library as a module "was so fast to include that it can barely be measured" ([s9w/cpp-lit](https://github.com/s9w/cpp-lit)).

### Standard headers and `std::ranges`

- Romeo, same benchmark: `<meta>` adds about 155 ms, `<print>` about 509 ms, and `<ranges>` about 86 ms per TU (310.2 ms with ranges vs 224.4 ms after rewriting to plain loops). His point: "Language features like templates are not the issue -- the Standard Library is." He builds about 900 TUs of his SFML fork in about 4.3 s by avoiding the standard library.
- Chromium measured adopting a subset of `std::views` on `base_unittests` and found results "very close", with a small but significant wall-time win (snippet: [chromium cxx thread](https://groups.google.com/a/chromium.org/g/cxx/c/rfMV-DxJ4Cg/m/MSWfr-AhEAAJ)).
- A May 2026 std-discussion benchmark found that the cost of including the standard library roughly doubled from C++11 to C++23 on GCC and Clang (snippet: [isocpp std-discussion](https://lists.isocpp.org/std-discussion/att-3369/README.md)).

### Concepts versus SFINAE

- Arthur O'Dwyer's generated benchmark (overload sets of 15 to 410 constrained candidates, Clang 14 trunk, GCC 10.3, MSVC 19.30): "On Clang, C++20 `requires` is significantly worse than the rest", GCC favours the extra-value-parameter SFINAE form, and MSVC behaves differently again. In part 2, a small rewrite made return-type SFINAE the worst on all three compilers. His conclusion: "in the average C++ codebase I don't think it indicates any need to avoid `requires` or even to prefer one kind of SFINAE over another" ([part 1](https://github.com/Quuxplusone/blog/blob/master/_posts/2021-09-04-enable-if-benchmark.md), [part 2](https://github.com/Quuxplusone/blog/blob/master/_posts/2021-09-14-enable-if-benchmark-part-2.md)). The same post explains libc++'s `_EnableIf` alias trick: it instantiates only two structs in total, where `enable_if` instantiates a new one for each use.
- Boost.Math reports shorter compile times after moving from SFINAE to concepts (snippet: [boostorg/math#848](https://github.com/boostorg/math/pull/848)). No measurement newer than 2021 compares the two forms in a controlled way.
- Compilers keep adding builtins for the hot metaprogramming paths. GCC 14: "New built-in `__type_pack_element` to speed up traits such as `std::tuple_element`" ([GCC 14 changes](https://gcc.gnu.org/gcc-14/changes.html)). GCC 15: "Compilation time speed ups, e.g. by improving hashing of template specializations" ([GCC 15 changes](https://gcc.gnu.org/gcc-15/changes.html)).

### xtensor, Eigen, Blaze, Kokkos

- xtensor: no published compile-time measurement found. The only related issues are a request for modules and concepts ([xtensor#2766](https://github.com/xtensor-stack/xtensor/issues/2766)), where the maintainer is "totally sure for using concepts as much as possible", and an arm64 CI build of `test_xblockwise_reducer.cpp` timing out ([xtensor#2770](https://github.com/xtensor-stack/xtensor/issues/2770)).
- Eigen: no published measurement found. An open tracker issue asks whether PCH, modules or `extern template` would help, with no measured answer (snippet: [eigen#1920](https://gitlab.com/libeigen/eigen/-/issues/1920)).
- Blaze: adding Blaze doubled one project's compile time. A bare include of `Blaze.h` took about 3 s, and including only the needed header (`CustomVector`) cut compile time by about 30% (snippet: [blaze#111](https://bitbucket.org/blaze-lib/blaze/issues/111/improve-compilation-time)).
- Kokkos: nvcc compile time explodes with `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK` ([kokkos#6325](https://github.com/kokkos/kokkos/issues/6325)). Kokkos builds for one GPU architecture per configuration, and fat binaries for several architectures are still an open request ([kokkos#4953](https://github.com/kokkos/kokkos/issues/4953)). Device compile work per build stays bounded, but each target GPU needs its own build.
- ITK reported about 15% shorter builds from `extern template` in first tests (snippet: [ITK discourse](https://discourse.itk.org/t/extern-templates-to-reduce-build-time/1105)).

## 4. Explicit instantiation and `extern template`

- What the standard allows: an explicit instantiation declaration (`extern template`) suppresses implicit instantiation of the definition, "except for inline functions and variables, declarations with types deduced from their initializer or return value, const variables of literal types, variables of reference types, and class template specializations" ([temp.explicit], [eel.is/c++draft/temp.explicit](https://eel.is/c++draft/temp.explicit)). In an expression-template library almost everything is inline or `auto`-returning, so the compiler may still instantiate it to inline it. The savings come only from non-inline, out-of-line functions.
- Kokkos core: ETI for deep copies and view fills was added in 2018 ([kokkos#1578](https://github.com/kokkos/kokkos/issues/1578)) and removed in 2020 ([kokkos#3157](https://github.com/kokkos/kokkos/issues/3157)). Christian Trott: "This ETI stuff was a largely failed experiment. It turns out in real applications it takes the compiler as much or more time to navigate the tree of partial specializations to find the right one, as it takes to compile the simple deep copy kernels" ([kokkos#3152](https://github.com/kokkos/kokkos/pull/3152)).
- Kokkos Kernels keeps ETI for its kernels, with a closed type list chosen at configure time (`KokkosKernels_INST_DOUBLE`, `_LAYOUTLEFT`, `_EXECSPACE_OPENMP`, `_OFFSET_INT`, ...). Each option says "Disabling this may increase build times". `KokkosKernels_ETI_ONLY` guarantees "that kernels are never built inside of object files which simply call KokkosKernels functions" ([BUILD.md](https://github.com/kokkos/kokkos-kernels/blob/develop/BUILD.md)).
- Lambdas and expression types cannot be named, so they cannot be explicitly instantiated. ETI fits only named functors or kernels over a known type grid.
- Modules give a similar effect without the boilerplate: an instantiation done inside a module interface is reused by every importer (the 20x instantiation case above). The limits are the same: only for types known when the module is built, and only on host compilers.

## 5. Type erasure at boundaries

No primary measurement isolates this technique. It follows from the measurements above: per-TU instantiation and optimization are what modules and ETI cannot remove, so the remaining lever is to stop the expression type from reaching every TU.

- PyTorch-style dynamic dispatch on dtype puts all kernels in a few `.cpp`/`.cu` files compiled once. User TUs see only a non-template tensor handle. The map rules out precompiled generic kernels, but the same boundary works when the kernel is generated at run time instead.
- In the run-time graph plus JIT family, user code builds graph nodes through a thin typed facade. The heavy templates (Kokkos kernels) are instantiated by the JIT, once per fused expression shape, and cached. Compile cost moves from the user's build to the first run.
- In the lighter-templates family, the cut can sit at assignment. The expression stays templated while it is built, then is lowered to a type-erased node list or to one of a small set of pre-instantiated evaluation kernels per (value type, rank, memory space). Fixed-shape tensors would stay fully templated because their size is the point.
- Dynamic-rank tensors are already a type-erasure boundary over rank. They are the natural ABI for bindings (see ticket 06).

## 6. Other techniques that work today

- Header hygiene: forward-declaration headers, no `<iostream>`/`<format>`/`<print>`/`<ranges>` in core headers, one header per feature (the Blaze 30% case, the Romeo numbers).
- PCH for tests, examples and benchmarks (`target_precompile_headers`). On GCC 16 it beats `import std` for small TUs. It also works with nvcc.
- Unity builds (`UNITY_BUILD`) for the test suite. CMake excludes sources scanned for modules from unity builds.
- Cheaper metaprogramming: prefer alias templates and `if constexpr` to class-template instantiation chains, prefer fold expressions to recursion, and use compiler builtins (`__type_pack_element`, `__make_integer_seq`) through the standard traits that wrap them.
- nvcc: `--threads` compiles several `-gencode` targets in parallel, and `--split-compile` parallelizes the optimizer. `--fdevice-time-trace` (CUDA 12.8+) writes a Chrome/Perfetto trace, and `ctadvisor` reads it and suggests those flags (snippet: [Compile Time Advisor](https://docs.nvidia.com/cuda/cuda-compile-time-advisor/index.html)). Building only for the target architecture matters as much as any C++ technique.
- Compile-time regression tests in CI. Kokkos opened such an effort early ([kokkos#1875](https://github.com/kokkos/kokkos/issues/1875)). Clang `-ftime-trace` plus ClangBuildAnalyzer, GCC `-ftime-report`, and MSVC Build Insights give per-template data.

## 7. Implications for tensorlib

1. The core headers must build without modules on every target compiler. A module interface (`tensorlib.cppm`) can be offered later behind an option, host backends only, following Kokkos `EXPERIMENTAL_CXX20_MODULES`. Do not depend on `import std` inside the core.
2. Use concepts for constraints, for diagnostics and readability. Compile time does not decide this.
3. Plan for ETI only on leaf containers and named evaluation kernels over a configurable type grid (Kokkos Kernels style). Expect nothing from it for expression trees.
4. The compile-time argument favours putting a type-erased boundary early (the JIT family, or lowering at assignment in the template family). This feeds ticket 03 (JIT with Kokkos) and the template-versus-graph decision.
5. Add a compile-time benchmark next to the runtime benchmarks: one TU per reference workload (stencil, multi-output fused expression, broadcast reduction), timed on GCC, Clang, nvcc and hipcc.

## 8. Gaps

- No primary source on hipcc or icpx module support either way. The table entries are inferences from Clang's offload issue and from CMake's compiler files.
- No controlled concepts-versus-SFINAE benchmark newer than 2021.
- No published compile-time numbers for xtensor or Eigen. A small benchmark on tensorlib's own reference workloads would be the reliable source.
