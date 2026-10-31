# CPU lowering of segments, gathers and scatters

Status: ready-for-agent
Blocked by: 23, 15, 16

## Context

Spec: [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation).
Decisions: [Lowering of short segments](../../core-architecture/issues/30-lowering-of-short-segments.md), [Kernel IR](../../core-architecture/issues/22-kernel-ir.md).
ADRs: [ADR 0009](../../../docs/adr/0009-evaluation-plan-and-per-element-ssa-kernel-ir.md).
Builds on: [LLVM code generator on CPU](23-llvm-cpu-generator.md), [Segment spaces and multi-body kernels](15-segment-spaces-and-multi-body-kernels.md), [Gathers, scatters and masked writes](16-gathers-scatters-and-masks.md).

## What to build

- Segmented kernels: struct-of-arrays table with one base column per distinct offset table, bridging of gaps up to 4 cells when declared, storage order kept, chunks balanced by cell count, LLVM's own tail.
- Several bodies selected by tag; gathers; affine scatter-add as gathers; atomic indirect scatter-add and the deterministic path.
- The 32-bit index variant behind the capability flag.

## Acceptance

- On the samurai interval dumps, a bridged kernel reaches the span-ceiling ratios of [Lowering of short segments](../../core-architecture/issues/30-lowering-of-short-segments.md) on the development VM (72 % or more on the advection mesh, one core).
- Bridging is never applied when gaps are not declared dead or bodies differ.
