# Compile-time budget in CI

Status: ready-for-agent
Blocked by: 02, 08

## Context

Spec: [Build, toolchains and packaging](../../core-architecture/spec.md#14-build-toolchains-and-packaging), [Testing and benchmarking](../../core-architecture/spec.md#16-testing-and-benchmarking).
Decisions: [Prototype of compile times for lighter templates](../../core-architecture/issues/11-compile-time-prototype.md), [Minimum C++ standard and build system](../../core-architecture/issues/14-cxx-standard-and-build.md), [Test and benchmark infrastructure](../../core-architecture/issues/32-test-and-benchmark-infrastructure.md).
Builds on: [Hosted CI matrix and header checks](02-ci-matrix-and-header-checks.md), [Contexts, pending windows and element-wise operations](08-contexts-windows-and-elementwise-operations.md).

## What to build

- A reference translation unit generated with 64 statements; Clang `-ftime-trace` measurements of template instantiations and front-end time; a threshold set at the first implementation plus 10 %.

## Acceptance

- The CI job fails when the budget is exceeded, with the top instantiations in the log.
