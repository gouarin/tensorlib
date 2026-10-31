# Hosted CI matrix and header checks

Status: resolved
Blocked by: 01

## Context

Spec: [Build, toolchains and packaging](../../core-architecture/spec.md#14-build-toolchains-and-packaging), [Style rules](../../core-architecture/spec.md#15-style-rules), [Testing and benchmarking](../../core-architecture/spec.md#16-testing-and-benchmarking).
Decisions: [Minimum C++ standard and build system](../../core-architecture/issues/14-cxx-standard-and-build.md), [Coding style and extension points](../../core-architecture/issues/29-coding-style-and-extension-points.md), [Test and benchmark infrastructure](../../core-architecture/issues/32-test-and-benchmark-infrastructure.md).
Builds on: [Build skeleton and repository layout](01-build-skeleton.md).

## What to build

- GitHub Actions on every PR: GCC 11.1 and latest, Clang 16 and latest on Linux; Apple Clang 15 and latest on macOS; MSVC 19.30 and latest on Windows.
- A serial-only build (no LLVM, no Kokkos) on every compiler; the Kokkos build is added by later tickets.
- ASan and UBSan jobs on the serial build.
- CI scripts that preprocess `<tl/tl.hpp>` and each component header and fail on a standard header outside the whitelist, on a C++23 conditional path in public headers, and on a public operation whose body is not one line (the facade rule of ADR 0011).
- A job that compiles and runs every documentation example on the serial backend (empty for now).

## Acceptance

- A test translation unit that includes `<source_location>` the way `tl::site` will compiles at every floor; the compile-only NVCC 12.2 and hipcc jobs of this ticket or of the GPU tickets report whether device compilers accept it in host code.
- A PR that includes `<algorithm>` from a public header fails the header check with a message naming the header and the include chain.
- All jobs are green on the skeleton.

## Answer

Done on branch `ci-matrix` (2026-10-31).

- `.github/workflows/ci.yml`, on every PR and on `main`:
  - Linux: GCC 11.1 (floor, `gcc:11.1` container), GCC 14, Clang 16 (floor) and Clang 18;
  - macOS: Apple Clang 15 (Xcode 15.4 on `macos-14`) and the latest stable Xcode on `macos-15`;
  - Windows: MSVC of the `windows-2022` image (latest);
  - ASan and UBSan (`linux-sanitize` preset, Clang 18);
  - install of a static and a shared build, then a consumer project built and run against it (`tests/install/`);
  - lint: clang-format 21, the public-header checks with GCC and Clang, the facade check and clang-tidy 21 with warnings as errors.

  Every build job runs `ctest -L "unit|example"`. GoogleTest comes through `TL_FETCH_DEPS=ON`.
- `tools/ci/check_public_headers.py` reads the include tree printed by `-H` and rejects a standard header outside the whitelist, naming the chain (`<tl/tl.hpp> -> <tl/detail/sort.hpp> -> <algorithm>`), and any `#if` on `__cplusplus`, `__cpp_*` or `__has_include` in public headers. Component headers are added to its `ENTRY_POINTS` table with the third-party prefixes they may include.
- `tools/ci/check_facades.py` reads Clang's JSON AST (Clang 16 or newer) and rejects a function template constrained by `tl::operand`, declared outside `tl/detail/`, whose body has more than one statement.
- Both checks have self-tests in CTest against fixtures in `tools/ci/tests/` (label `unit`).
- Documentation examples live in `examples/`, one executable test each with the label `example`; `examples/version.cpp` is the first.
- `tests/core/toolchain.cpp` checks that a defaulted `std::source_location::current()` captures the caller, as `tl::site` will.

Replayed locally: the GCC 11.1 job in the `gcc:11.1` container and the Clang 16 job in `silkeh/clang:16` (all tests pass), the lint steps with clang 21, the sanitizer preset, the install check, and `actionlint` on the workflow. Not run: macOS, Windows and the workflow itself on GitHub, since the repository has no remote yet.

Findings:

- Clang 15 captures the wrong site: a defaulted `std::source_location::current()` gives the line of the default argument, not the caller's (Clang 15.0.7; Clang 16.0.6 is right). The host floor becomes Clang 16 (amendment on the floor in [Minimum C++ standard and build system](../../core-architecture/issues/14-cxx-standard-and-build.md)). Clang 15's JSON AST also lacks the node the facade check relies on.
- NVCC 12.2 accepts `std::source_location` in host code and captures the right line.
- The MSVC 19.30 floor is not installed on the hosted `windows-2022` image, so only the latest MSVC runs. A floor job needs a self-hosted runner or a toolset that the image does not ship.
- hipcc and the SYCL compiler are not checked yet; their compile-only jobs come with the Kokkos and GPU tickets.
