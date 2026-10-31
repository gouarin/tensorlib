# CPU lowering of reductions, scans and fixed lanes

Status: ready-for-agent
Blocked by: 23, 13

## Context

Spec: [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation).
Decisions: [Code-generation strategy](../../core-architecture/issues/12-code-generation-strategy.md), [Kernel IR](../../core-architecture/issues/22-kernel-ir.md), [Prototype of the LLVM JIT on CPU](../../core-architecture/issues/27-llvm-jit-cpu-prototype.md).
ADRs: [ADR 0009](../../../docs/adr/0009-evaluation-plan-and-per-element-ssa-kernel-ir.md).
Builds on: [LLVM code generator on CPU](23-llvm-cpu-generator.md), [Reductions and scans](13-reductions-and-scans.md).

## What to build

- Canonical-block reductions emitted explicitly and combined from `parallel_reduce` in a fixed order; scans.
- Lane-typed `fixed` values with component strides, scalarised for LLVM, counted by the ceilings.
- Traced combiners and `call` to external functions (bitcode inlined when present, host symbol otherwise).

## Acceptance

- Reductions give the same bits at 1, 4 and 16 threads.
- The canonical sum is faster than a Kokkos `parallel_reduce` on one thread (prototype: 4 to 6 times).
