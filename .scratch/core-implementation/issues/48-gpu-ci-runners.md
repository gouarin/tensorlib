# Self-hosted GPU runners

Status: ready-for-human

## Context

Spec: [Testing and benchmarking](../../core-architecture/spec.md#16-testing-and-benchmarking).
Decisions: [Test and benchmark infrastructure](../../core-architecture/issues/32-test-and-benchmark-infrastructure.md).

## What to build

- Register at least one NVIDIA machine (AMD if possible) as a self-hosted GitHub runner; nightly GPU jobs and the `gpu` label on PRs.
- Arrange periodic campaigns in computing centres (GitLab CI with Jacamar or batch jobs) for multi-node MPI, AMD and Intel GPUs.

## Acceptance

- A trivial CUDA job runs on the runner from a PR labelled `gpu`.
