# Traced functions, traced combiners and external functions

Status: ready-for-agent
Blocked by: 13

## Context

Spec: [Evaluation](../../core-architecture/spec.md#5-evaluation), [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation), [Extension points](../../core-architecture/spec.md#12-extension-points).
Decisions: [Reconcile the core with samurai and ponio](../../core-architecture/issues/26-reconcile-with-samurai-ponio.md), [Kernel IR](../../core-architecture/issues/22-kernel-ir.md), [Coding style and extension points](../../core-architecture/issues/29-coding-style-and-extension-points.md).
ADRs: [ADR 0011](../../../docs/adr/0011-thin-templates-over-an-erased-core-and-layered-extension-points.md).
Builds on: [Reductions and scans](13-reductions-and-scans.md).

## What to build

- `tl::apply` tracing a generic lambda once on symbolic arguments; recorded operations fuse with the window; `fixed` values work in traced code and eagerly.
- `tl::reduce(e, init, combine)` with a traced associative combiner over a `fixed` state; associativity checked in debug mode by comparing two splits.
- The external-function registry and the IR `call` operation: required host symbol, optional LLVM bitcode per target (stored, used by the generator later), bitcode hash in the key; the serial interpreter calls the host symbol.

## Acceptance

- A Welford mean and variance written as a traced combiner matches a reference.
- A Rusanov flux written as a generic lambda fuses with the surrounding window into one kernel.
