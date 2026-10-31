# Exceptions, poison and finalize

Status: ready-for-agent
Blocked by: 09

## Context

Spec: [Errors and debugging](../../core-architecture/spec.md#10-errors-and-debugging).
Decisions: [Error reporting and debug mode](../../core-architecture/issues/24-error-reporting-and-debug-mode.md), [Minimum C++ standard and build system](../../core-architecture/issues/14-cxx-standard-and-build.md), [Buffer reuse and allocation](../../core-architecture/issues/21-buffer-reuse-and-allocation.md).
ADRs: [ADR 0007](../../../docs/adr/0007-errors-poison-results-across-ranks.md).
Builds on: [Planner: fusing a window into kernels](09-window-fusion-planner.md).

## What to build

- The exception hierarchy (`tl::shape_error`, `tl::unsupported_operation`, `tl::device_error`, `tl::out_of_memory`, `tl::spmd_error`, `tl::remote_error`).
- Poisoned tensors: a failed kernel poisons its outputs, poison propagates, reading throws; `tl::wait()` and `tl::eval_all()` throw the first unreported error of the context; `fatal()`.
- `tl::initialize` and `tl::finalize`: unread errors and live tensors reported with sites and rank.
- Host usage errors thrown at once with the site.

## Acceptance

- A kernel failing on the serial backend poisons only the tensors that depend on it.
- A live tensor at `tl::finalize` is listed with its site.
