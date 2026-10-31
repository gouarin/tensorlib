# Redistribution

Status: ready-for-agent
Blocked by: 34

## Context

Spec: [Distributions and communication](../../core-architecture/spec.md#9-distributions-and-communication).
Decisions: [Communication in the evaluation plan](../../core-architecture/issues/28-communication-in-the-evaluation-plan.md).
ADRs: [ADR 0010](../../../docs/adr/0010-persistent-communication-patterns-and-a-progress-thread.md).
Builds on: [Ghost exchanges in the plan](34-ghost-exchanges-in-the-plan.md).

## What to build

- `redistribute()` and `replicate()` as non-blocking plan nodes for the four v1 cases, with patterns cached per pair of distributions.

## Acceptance

- Each case round-trips data exactly on 1 to 4 thread ranks.
