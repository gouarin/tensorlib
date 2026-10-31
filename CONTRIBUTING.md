# Contributing to tensorlib

The design of the library is in [the architecture spec](.scratch/core-architecture/spec.md), its vocabulary in [`CONTEXT.md`](CONTEXT.md) and its decisions in [`docs/adr/`](docs/adr/). Work is tracked as tickets in [`.scratch/core-implementation/`](.scratch/core-implementation/spec.md).

## Build and test

You need CMake 3.25.2 or newer, Ninja and a C++20 compiler (GCC 11.1, Clang 15, Apple Clang 15 or MSVC 19.30 and newer). Tests use GoogleTest, found with `find_package`. If it is not installed, add `-DTL_FETCH_DEPS=ON` and CMake downloads it.

```bash
cmake --preset linux-debug          # or macos-debug, windows-debug
cmake --build --preset linux-debug
ctest --preset linux-debug -L unit
```

The `*-ci` presets turn warnings into errors, as CI does. CTest labels select groups of tests: `unit`, `generated`, `mpi`, `gpu`, `bench`, `abi` and `array_api`.

Before you open a pull request, run `clang-format` and `clang-tidy` (version 21) on the files you changed:

```bash
clang-format -i <files>
clang-tidy -p build/linux-debug <files>
```

## Style rules

The full rules are in [section 15 of the spec](.scratch/core-architecture/spec.md#15-style-rules). Tools check most of them: clang-format, clang-tidy, compiler warnings, and CI scripts for the public header whitelist and the one-line facade rule. A reviewer checks the rest with this list:

- Names come from the glossary in `CONTEXT.md`. A new concept gets a glossary entry before it gets a type.
- There is one way to write each thing: no second spelling of an operation, no convenience member next to a free function.
- Extensions use the mechanisms of [ADR 0011](docs/adr/0011-thin-templates-over-an-erased-core-and-layered-extension-points.md): traits for element types, run-time registries for behaviour, compositions for new operations.
- A public function reads without opening its `.cpp`: its name, its parameters and its documentation example say what it does.
- Every public function has a documentation example that CI compiles and runs on the serial backend.
- Every fixed bug comes with a regression test.

## Commits

Write commit messages in English, in the imperative, with a short first line. One logical change per commit.
