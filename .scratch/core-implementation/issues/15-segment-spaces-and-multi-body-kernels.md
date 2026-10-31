# Segment spaces and multi-body kernels

Status: ready-for-agent
Blocked by: 11

## Context

Spec: [Tensors, views and expressions](../../core-architecture/spec.md#3-tensors-views-and-expressions), [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation).
Decisions: [Views and slicing](../../core-architecture/issues/23-views-and-slicing.md), [Reconcile the core with samurai and ponio](../../core-architecture/issues/26-reconcile-with-samurai-ponio.md), [Kernel IR](../../core-architecture/issues/22-kernel-ir.md), [Lowering of short segments](../../core-architecture/issues/30-lowering-of-short-segments.md).
ADRs: [ADR 0009](../../../docs/adr/0009-evaluation-plan-and-per-element-ssa-kernel-ir.md).
Builds on: [Affine views and slicing](11-affine-views-and-slicing.md).

## What to build

- Segmented iteration spaces in the IR: 1-d segments with trailing ordinary axes, one body tag per segment, one base offset per segment per offset table, tables as arguments.
- `tl::segment_space S(lengths)` and `S.on(offsets, stride)` views, congruent by identity; declaration of dead, accessible gaps (for bridging).
- Several bodies per kernel selected by segment tag.
- Partial writes on a shared buffer: one fused detach per owner per window, regions as segments.
- Serial interpreter support.

## Acceptance

- A samurai-like update over interval tables (the dumps of [Lowering of short segments](../../core-architecture/issues/30-lowering-of-short-segments.md)) gives the same result as a hand loop.
- Interior and boundary schemes run in one kernel with two bodies.
- Views from different segment spaces are an error at node build.
