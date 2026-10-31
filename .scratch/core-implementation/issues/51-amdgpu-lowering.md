# AMD lowering through AMDGPU

Status: ready-for-agent
Blocked by: 50

## Context

Spec: [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation), [Backends](../../core-architecture/spec.md#7-backends).
Decisions: [Code-generation strategy](../../core-architecture/issues/12-code-generation-strategy.md), [Rerun the fusion benchmark on an HPC node and a GPU](../../core-architecture/issues/19-rerun-fusion-benchmark-hpc-gpu.md).
ADRs: [ADR 0008](../../../docs/adr/0008-in-process-llvm-as-the-only-code-generator.md).
Builds on: [NVIDIA lowering through NVPTX](50-nvptx-lowering.md).

## What to build

- AMDGPU code generation and HIP launch, with the same intrinsics table entries.

## Acceptance

- The window generator passes on an AMD GPU.
