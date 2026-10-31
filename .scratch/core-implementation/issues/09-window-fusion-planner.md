# Planner: fusing a window into kernels

Status: ready-for-agent
Blocked by: 08

## Context

Spec: [Evaluation](../../core-architecture/spec.md#5-evaluation), [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation), [Memory](../../core-architecture/spec.md#8-memory).
Decisions: [When the task graph is evaluated](../../core-architecture/issues/08-evaluation-trigger.md), [Prototype of multi-output fusion](../../core-architecture/issues/09-multi-output-fusion-prototype.md), [Buffer reuse and allocation](../../core-architecture/issues/21-buffer-reuse-and-allocation.md), [Kernel IR](../../core-architecture/issues/22-kernel-ir.md).
ADRs: [ADR 0002](../../../docs/adr/0002-implicit-pending-windows.md), [ADR 0009](../../../docs/adr/0009-evaluation-plan-and-per-element-ssa-kernel-ir.md).
Builds on: [Contexts, pending windows and element-wise operations](08-contexts-windows-and-elementwise-operations.md).

## What to build

- The evaluation plan: fuse all live outputs of a window into one kernel with several stores when shapes allow, inline expressions, drop dead outputs (atomic dead-output flag).
- Buffer plan: in-place reuse for assignment targets with the capacity rule (fits and fills at least half), checked once per buffer on the host; temporaries allocated at first writer and freed after last reader.
- `resize(shape)` as `t = tl::empty<T>(shape)`.
- The deterministic resource-limit triggers (node count, planned bytes).

## Acceptance

- `c = a + b; d = a + 2*b` is one kernel (checked from the plan dump).
- A uniquely held target reuses its buffer; a shared one does not.
- Results match the unfused evaluation bit for bit on the serial backend.
