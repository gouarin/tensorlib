# NVIDIA lowering through NVPTX

Status: ready-for-agent
Blocked by: 25, 26, 48

## Context

Spec: [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation), [Backends](../../core-architecture/spec.md#7-backends).
Decisions: [Code-generation strategy](../../core-architecture/issues/12-code-generation-strategy.md), [Rerun the fusion benchmark on an HPC node and a GPU](../../core-architecture/issues/19-rerun-fusion-benchmark-hpc-gpu.md).
ADRs: [ADR 0008](../../../docs/adr/0008-in-process-llvm-as-the-only-code-generator.md).
Builds on: [CPU lowering of reductions, scans and fixed lanes](25-cpu-reductions-scans-and-fixed-lanes.md), [CPU lowering of segments, gathers and scatters](26-cpu-segments-gathers-and-scatters.md), [Self-hosted GPU runners](48-gpu-ci-runners.md).

## What to build

- NVPTX code generation, driver PTX JIT, launch on the stream of the Kokkos execution-space instance with a grid from capabilities; intrinsics table (barriers, shared memory, atomics); stencil tiling from shifts; GPU kernel-size ceilings.
- The CUDA rows of [Rerun the fusion benchmark on an HPC node and a GPU](../../core-architecture/issues/19-rerun-fusion-benchmark-hpc-gpu.md).

## Acceptance

- The window generator passes on an NVIDIA GPU against the serial backend, with ulp bounds for transcendental functions.
