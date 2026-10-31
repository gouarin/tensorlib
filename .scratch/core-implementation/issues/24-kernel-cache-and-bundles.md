# Kernel cache and kernel bundles

Status: ready-for-agent
Blocked by: 23

## Context

Spec: [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation).
Decisions: [Code-generation strategy](../../core-architecture/issues/12-code-generation-strategy.md).
ADRs: [ADR 0008](../../../docs/adr/0008-in-process-llvm-as-the-only-code-generator.md).
Builds on: [LLVM code generator on CPU](23-llvm-cpu-generator.md).

## What to build

- The cache key of [Code-generation strategy](../../core-architecture/issues/12-code-generation-strategy.md) and a memory then disk cache (`$XDG_CACHE_HOME/tensorlib`, `TL_CACHE_DIR`); only local rank 0 of a node writes, through atomic rename; unwritable directory falls back to memory.
- Kernel bundles produced by a training run or by `tl-aot` from a recorded trace, loaded read-only; `TL_JIT=off`.

## Acceptance

- Reloading a kernel from disk costs about 0.06 ms (prototype figure, order of magnitude).
- Under `TL_JIT=off`, a kernel missing from the bundle raises an error naming it.
