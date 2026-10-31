# Hosted CI matrix and header checks

Status: ready-for-agent
Blocked by: 01

## Context

Spec: [Build, toolchains and packaging](../../core-architecture/spec.md#14-build-toolchains-and-packaging), [Style rules](../../core-architecture/spec.md#15-style-rules), [Testing and benchmarking](../../core-architecture/spec.md#16-testing-and-benchmarking).
Decisions: [Minimum C++ standard and build system](../../core-architecture/issues/14-cxx-standard-and-build.md), [Coding style and extension points](../../core-architecture/issues/29-coding-style-and-extension-points.md), [Test and benchmark infrastructure](../../core-architecture/issues/32-test-and-benchmark-infrastructure.md).
Builds on: [Build skeleton and repository layout](01-build-skeleton.md).

## What to build

- GitHub Actions on every PR: GCC 11.1 and latest, Clang 15 and latest on Linux; Apple Clang 15 and latest on macOS; MSVC 19.30 and latest on Windows.
- A serial-only build (no LLVM, no Kokkos) on every compiler; the Kokkos build is added by later tickets.
- ASan and UBSan jobs on the serial build.
- CI scripts that preprocess `<tl/tl.hpp>` and each component header and fail on a standard header outside the whitelist, on a C++23 conditional path in public headers, and on a public operation whose body is not one line (the facade rule of ADR 0011).
- A job that compiles and runs every documentation example on the serial backend (empty for now).

## Acceptance

- A test translation unit that includes `<source_location>` the way `tl::site` will compiles at every floor; the compile-only NVCC 12.2 and hipcc jobs of this ticket or of the GPU tickets report whether device compilers accept it in host code.
- A PR that includes `<algorithm>` from a public header fails the header check with a message naming the header and the include chain.
- All jobs are green on the skeleton.
