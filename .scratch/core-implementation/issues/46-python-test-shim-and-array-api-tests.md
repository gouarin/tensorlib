# Python test shim and array-api-tests

Status: ready-for-agent
Blocked by: 44, 45

## Context

Spec: [C ABI and bindings](../../core-architecture/spec.md#13-c-abi-and-bindings), [Testing and benchmarking](../../core-architecture/spec.md#16-testing-and-benchmarking).
Decisions: [Test and benchmark infrastructure](../../core-architecture/issues/32-test-and-benchmark-infrastructure.md), [C ABI and binding surface](../../core-architecture/issues/31-c-abi-and-binding-surface.md).
Builds on: [C ABI types, distributions and extensions](44-c-abi-types-distributions-and-extensions.md), [DLPack import and export](45-c-abi-dlpack.md).

## What to build

- A test-only Python namespace over `<tl/c.h>` (ctypes or nanobind), generated from the catalogue, enough to run the official array-api-tests suite and comparisons with NumPy in CI (`array_api` label). Not a supported binding.

## Acceptance

- array-api-tests runs in CI with a documented list of expected failures, each linked to a ticket.
