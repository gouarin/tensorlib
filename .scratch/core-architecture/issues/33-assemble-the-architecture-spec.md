# Assemble the architecture spec

Type: task
Status: resolved
Blocked by: 29, 30, 31, 32

## Question

Write the destination: the architecture spec of the core, from the resolved tickets, `CONTEXT.md` and the ADRs, ready to be split into implementation tickets. It links each decision to its ticket instead of restating its reasoning. It states the style rules of [Coding style and extension points](29-coding-style-and-extension-points.md) and lists what remains open: the hardware checks of [Rerun the fusion benchmark on an HPC node and a GPU](19-rerun-fusion-benchmark-hpc-gpu.md) and any fog left on the map. The user reviews it before the map closes.

## Answer

Done on 2026-10-31. The spec is [spec.md](../spec.md): sixteen sections by component, each linking to its tickets and ADRs, the style rules written out, nine reconciliations between tickets amended by later ones, the open items (hardware checks of [Rerun the fusion benchmark on an HPC node and a GPU](19-rerun-fusion-benchmark-hpc-gpu.md), the fog on device communication, unmeasured cases, infrastructure, defaults to set) and a suggested order of work.

The user approved two points that go slightly beyond the tickets:

- the default queue belongs to a pair of context and device, following [ADR 0012](../../../docs/adr/0012-one-c-abi-with-explicit-contexts-and-generic-dispatch.md) (reconciliation 3);
- a segment table has one base column per distinct offset table, and bridging applies only when every operand shares the same table (reconciliation 9).

While assembling, the CMake target of the C ABI was corrected from `tl::c` to `tensorlib::c` in [C ABI and binding surface](31-c-abi-and-binding-surface.md), ADR 0012 and the map, to follow [Minimum C++ standard and build system](14-cxx-standard-and-build.md).

## Comments

2026-10-31: draft written in [spec.md](../spec.md), awaiting the user's review before the map closes.

2026-10-31: the user approved the spec and the two points above; ticket and map closed.
