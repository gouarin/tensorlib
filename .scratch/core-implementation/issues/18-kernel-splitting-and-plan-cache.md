# Kernel splitting and the plan cache

Status: ready-for-agent
Blocked by: 09

## Context

Spec: [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation).
Decisions: [Code-generation strategy](../../core-architecture/issues/12-code-generation-strategy.md), [Prototype of multi-output fusion](../../core-architecture/issues/09-multi-output-fusion-prototype.md).
ADRs: [ADR 0008](../../../docs/adr/0008-in-process-llvm-as-the-only-code-generator.md).
Builds on: [Planner: fusing a window into kernels](09-window-fusion-planner.md).

## What to build

- Splitting a fused kernel by `TL_MAX_KERNEL_OPS` and `TL_MAX_KERNEL_LIVE` (lanes counted), cutting where the stored temporary is smallest; deterministic.
- The automatic plan cache keyed by the window signature (structure key, buffer identity, aliasing).
- `TL_DEBUG=dump` printing the plan and the kernel IR.

## Acceptance

- A ponio-like 20-stage combination stays one kernel; a long chain splits at the cheapest point.
- A window seen before skips planning (measured: hashing under 1 us).
