# LLVM code generator on CPU

Status: ready-for-agent
Blocked by: 22

## Context

Spec: [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation).
Decisions: [Code-generation strategy](../../core-architecture/issues/12-code-generation-strategy.md), [Prototype of the LLVM JIT on CPU](../../core-architecture/issues/27-llvm-jit-cpu-prototype.md).
ADRs: [ADR 0008](../../../docs/adr/0008-in-process-llvm-as-the-only-code-generator.md).
Builds on: [Kokkos backend skeleton](22-kokkos-backend-skeleton.md).

## What to build

- Lowering of rectangular element-wise kernels to LLVM IR, ORC JIT, `body(begin, end, args)` called by a precompiled trampoline from `Kokkos::parallel_for`; scalar inner loop with `noalias` and vectorisation metadata; peeled head loop for alignment.
- FMA contraction allowed; `TL_FP=strict` turns it off; no fast-math on NaN and Inf.
- Kernels named after their fused sites; evaluations as Kokkos Tools regions.
- LLVM required in the Kokkos backend; Kokkos plus LLVM builds in CI (Linux, macOS, Windows).

## Acceptance

- Generated kernels match the serial backend bit for bit under `TL_FP=strict` across the window generator.
- ponio20 runs at the machine's bandwidth ceiling, as in [Prototype of the LLVM JIT on CPU](../../core-architecture/issues/27-llvm-jit-cpu-prototype.md).
