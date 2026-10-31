# Gathers, scatters and masked writes

Status: ready-for-agent
Blocked by: 13, 15

## Context

Spec: [Tensors, views and expressions](../../core-architecture/spec.md#3-tensors-views-and-expressions), [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation).
Decisions: [Views and slicing](../../core-architecture/issues/23-views-and-slicing.md), [Kernel IR](../../core-architecture/issues/22-kernel-ir.md).
ADRs: [ADR 0006](../../../docs/adr/0006-views-are-references-distinct-from-tensors.md).
Builds on: [Reductions and scans](13-reductions-and-scans.md), [Segment spaces and multi-body kernels](15-segment-spaces-and-multi-body-kernels.md).

## What to build

- `u(idx)` as a fusible gather; `u(idx) = e` as a scatter; `u(idx) += e` and `-=` accumulating duplicates.
- Affine scatter-adds turned into gathers; indirect scatter-add atomic by default, and a deterministic sort-based path (cached per `idx` version) under `TL_FP=strict` or `tl::deterministic(idx)`.
- `u(mask) = e` as an in-place `where`; `tl::compress(u, mask)` as a non-fusible trigger; `tl::where`.

## Acceptance

- `u(I) -= f; u(I + 1) += f;` lowers to a gather (checked in the IR dump).
- The deterministic path gives the same bits at every thread count.
- Reading `u(mask)` does not compile.
