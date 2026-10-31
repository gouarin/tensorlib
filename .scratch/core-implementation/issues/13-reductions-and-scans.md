# Reductions and scans

Status: ready-for-agent
Blocked by: 09

## Context

Spec: [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation).
Decisions: [Kernel IR](../../core-architecture/issues/22-kernel-ir.md), [Code-generation strategy](../../core-architecture/issues/12-code-generation-strategy.md).
ADRs: [ADR 0009](../../../docs/adr/0009-evaluation-plan-and-per-element-ssa-kernel-ir.md).
Builds on: [Planner: fusing a window into kernels](09-window-fusion-planner.md).

## What to build

- Reductions as kernel outputs in the IR: built-in combiners (`sum`, `prod`, `min`, `max`, `all`, `any`, `argmin`, `argmax`), reduced axes, identities, several reductions fused with stores over one space.
- Canonical blocks independent of thread count and SIMD width, combined in a fixed tree.
- Array API functions with option aggregates (`tl::sum(a, {.axis = 0, .keepdims = true})`).
- Scans as kernels of their own with a fused prologue (`cumulative_sum`).

## Acceptance

- Reductions on empty tensors return the identity.
- A reduction fused with a store is one kernel.
- The canonical order gives the same bits as a reference implementation of the block scheme.
