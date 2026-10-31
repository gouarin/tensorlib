# Kernel IR for rectangular kernels

Status: ready-for-agent
Blocked by: 04

## Context

Spec: [Evaluation plan, kernel IR and code generation](../../core-architecture/spec.md#6-evaluation-plan-kernel-ir-and-code-generation).
Decisions: [Kernel IR](../../core-architecture/issues/22-kernel-ir.md), [Code-generation strategy](../../core-architecture/issues/12-code-generation-strategy.md).
ADRs: [ADR 0009](../../../docs/adr/0009-evaluation-plan-and-per-element-ssa-kernel-ir.md).
Builds on: [Shapes, dtypes and Array API type promotion](04-shapes-dtypes-and-promotion.md).

## What to build

- Per-element SSA program over an n-d rectangular iteration space: loads through an affine index map (offset, strides, structural stride-0 and inner-stride flags, stencil shifts), arguments (scalars), element-wise operations, explicit `cast`, `select`, stores.
- Separation of structure (part of the key) from arguments (extents, offsets, stride values, scalars).
- Structural hash computed incrementally while building; versioned text form that prints and parses; binary serialisation.
- The `tl-opt` tool and FileCheck tests on text files.
- Each operation carries a site index.

## Acceptance

- `parse(print(ir)) == ir` and hash stability over the test corpus.
- Two kernels that differ only in extents or scalar values hash the same.

## Not in this ticket

- Reductions, scans, segments, gathers, scatters, several bodies, `call` (later tickets extend the IR).
