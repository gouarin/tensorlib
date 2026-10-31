# Build skeleton and repository layout

Status: resolved

## Context

Spec: [Build, toolchains and packaging](../../core-architecture/spec.md#14-build-toolchains-and-packaging), [Style rules](../../core-architecture/spec.md#15-style-rules).
Decisions: [Minimum C++ standard and build system](../../core-architecture/issues/14-cxx-standard-and-build.md), [Coding style and extension points](../../core-architecture/issues/29-coding-style-and-extension-points.md).

## What to build

- CMake project (3.25.2 or newer) with `CMakePresets.json` for Linux, macOS and Windows, C++20, and `find_package(tensorlib)` exporting `tensorlib::core`.
- Dependencies through `find_package` only; `TL_FETCH_DEPS=ON` turns on `FetchContent` for developers; nothing vendored.
- Layout: `include/tl/` (public), `include/tl/detail/` (internal), `src/<component>/` mirroring it, umbrella header `<tl/tl.hpp>`, `tests/` mirroring `src/`.
- `.clang-format`, the curated `.clang-tidy` profile (`modernize`, `readability`, `bugprone`, `performance`, part of `cppcoreguidelines`, cognitive-complexity threshold), and strict warnings (`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`, `/W4`) as errors in CI only, never on installed targets.
- GoogleTest through CTest, with the labels `unit`, `generated`, `mpi`, `gpu`, `bench`, `abi`, `array_api`.
- `CONTRIBUTING.md` with the review checklist of the style rules, and the `TL_ENABLE_ASSERTIONS` CMake option.

## Acceptance

- A consumer project builds against the installed package with `find_package(tensorlib)` and `target_link_libraries(... tensorlib::core)`.
- `ctest -L unit` runs an empty GoogleTest suite on all three platforms.
- clang-format and clang-tidy pass on the tree.

## Not in this ticket

- CI workflows (next ticket).
- Any tensor code.

## Answer

Done on branch `build-skeleton` (2026-10-31).

- `CMakeLists.txt`, `cmake/` and `CMakePresets.json`: C++20, CMake 3.25.2 or newer, presets `{linux,macos,windows}-{debug,release,ci}` (debug turns on `TL_ENABLE_ASSERTIONS`, ci turns warnings into errors), options `TL_BUILD_TESTS`, `TL_FETCH_DEPS`, `TL_WARNINGS_AS_ERRORS`, `TL_ENABLE_ASSERTIONS`. Strict warnings are `PRIVATE` to tensorlib's targets (`tl_set_warnings`).
- `find_package(tensorlib)` exports `tensorlib::core` (library `libtensorlib`, `SameMinorVersion` before 1.0). Windows DLLs export every symbol, since the library exports functions only.
- The only code is the version: `<tl/version.hpp>`, configured by CMake, with `tl::version`, `tl::header_version` and `tl::library_version()` to detect headers and library from different installations. `<tl/tl.hpp>` is the umbrella header (IWYU export pragma).
- Tests: GoogleTest through `tl_add_test(name SOURCES ... LABELS ...)` and `gtest_discover_tests`; `tests/install/` is a consumer project for the CI install check.
- `.clang-format` is samurai's (Mozilla base, Allman braces, 4 spaces, 140 columns). `.clang-tidy` is the curated profile with naming rules and a cognitive-complexity threshold of 15.
- `CONTRIBUTING.md` holds the build steps and the review checklist.

Checked on the sandbox (Linux aarch64): GCC 15.2 and Clang 21.1 with warnings as errors, `ctest -L unit` passes; a consumer project builds and runs against the installed package, static and shared; `TL_FETCH_DEPS=ON` downloads GoogleTest 1.17 and passes; clang-format and clang-tidy 21 report nothing. macOS and Windows are left to the CI matrix of [Hosted CI matrix and header checks](02-ci-matrix-and-header-checks.md).
