# Techniques to cut compile times of template-heavy libraries

Type: research
Status: resolved

## Question

What works in 2026 to keep compile times of expression-template libraries reasonable? Cover the real status of C++20 modules and `import std` on GCC, Clang, Apple Clang, MSVC, nvcc, hipcc and icpx, explicit instantiation, `extern template`, type erasure at boundaries, concepts versus SFINAE, and published measurements (xtensor, Eigen, Blaze, Kokkos, std::ranges).

## Answer

- Modules cannot be the strategy. nvcc does not support them (host or device), no GPU offload compiler does, Apple Clang has no `import std` or CMake scanning, and GCC 16 still calls them experimental. The core must build as plain headers. An optional host-only `import tensorlib` wrapper is possible, as Kokkos 5 does it. Even where modules work, they save parsing, not per-TU instantiation: real projects gain 10 to 50%, and PCH beats `import std` on GCC 16 for small TUs.
- `extern template` and ETI only help for a closed, nameable type grid (leaf containers, named kernels; Kokkos Kernels style). Kokkos core dropped its ETI as "a largely failed experiment". They do nothing for expression trees or lambdas.
- Concepts versus SFINAE: pick concepts for diagnostics. The measured compile-time differences depend on the compiler and are small in real code.
- The real levers: no heavy std headers in core headers (`<ranges>` costs about 86 ms per TU and `<print>` about 509 ms on GCC 16), shallow instantiation, a type-erased boundary early (JIT graph, or lowering at assignment), and compile-time benchmarks in CI.

Findings: [research/02-compile-time-techniques.md](../research/02-compile-time-techniques.md)
