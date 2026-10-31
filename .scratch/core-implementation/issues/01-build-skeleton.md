# Build skeleton and repository layout

Status: ready-for-agent

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
