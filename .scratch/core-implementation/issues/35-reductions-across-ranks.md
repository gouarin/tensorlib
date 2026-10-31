# Reductions across ranks

Status: ready-for-agent
Blocked by: 34, 13

## Context

Spec: [Distributions and communication](../../core-architecture/spec.md#9-distributions-and-communication).
Decisions: [Communication in the evaluation plan](../../core-architecture/issues/28-communication-in-the-evaluation-plan.md).
ADRs: [ADR 0010](../../../docs/adr/0010-persistent-communication-patterns-and-a-progress-thread.md).
Builds on: [Ghost exchanges in the plan](34-ghost-exchanges-in-the-plan.md), [Reductions and scans](13-reductions-and-scans.md).

## What to build

- One packed non-blocking reduction per exchange point with the poison word and the plan hash; generated combining operation (traced combiners included); completion as an event so a 0-d result stays on the device.
- The fixed-shape tree in rank order under `TL_FP=strict` or `tl::deterministic(...)`.

## Acceptance

- Under `TL_FP=strict`, every rank gets the same bits at a fixed rank count, run after run.
- Empty local parts contribute the identity.
