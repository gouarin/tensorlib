# Lowering of short segments

Type: prototype
Status: resolved

## Question

[Prototype of the LLVM JIT on CPU](27-llvm-jit-cpu-prototype.md) found that segmented kernels on short samurai segments (8 to 64 cells) run at 60 to 82 % of a flat stencil. Hand-written loops pay the same cost, so the gap comes from the segment loop itself, not from the JIT. Find a lowering of the segment space of [Kernel IR](22-kernel-ir.md) that closes most of that gap.

Starting from the prototype in `prototypes/27-llvm-jit-cpu/`, compare at least:

- merging adjacent segments whose offsets continue one another, once at plan time;
- predicated or masked tails, so a short segment runs as whole vectors;
- a segment table laid out for the lowering (struct of arrays, sorted by length, length classes with one body each);
- several segments per thread chunk against one segment per iteration.

Use interval layouts taken from real samurai meshes (`advection_2d` with adaptation, the LBM Rayleigh-Taylor case), not synthetic lengths only. The result says which lowering the generator uses, what it requires of `tl::segment_space`, and whether anything must change on the samurai side.

## Answer

The prototype ([README](../prototypes/30-short-segments/README.md)) ran on two real samurai meshes: `advection_2d` at level 13 (67 815 intervals, 782 374 cells) and the LBM Rayleigh-Taylor case at level 9 (5 663 intervals, 121 336 cells). Every variant gives the same bits as the lowering of [Prototype of the LLVM JIT on CPU](27-llvm-jit-cpu-prototype.md). Measured on the 18-vCPU aarch64 VM (NEON, no SVE), at 1 and 16 threads; read the numbers as orders of magnitude.

What real meshes look like:

- The median interval has 4 cells, not 8 to 64. On the advection mesh, prototype 27's lowering runs at 49 % of the reachable ceiling on one core.
- samurai stores ghost cells between intervals (gaps of 1 to 4 cells), so no interval ever continues the previous one in storage. Merging contiguous segments never applies.
- A third of the storage span is ghosts (span / cells = 1.5). The reachable ceiling is a flat loop over the span.

Decision (user, 2026-10-31): the generator **bridges** segments declared bridgeable by `tl::segment_space`.

- At plan time, two consecutive segments whose storage gap is at most a maximum gap (4 cells by default) are merged, and the kernel computes the gap cells too. This took the advection mesh from 49 % to 72-78 % of the span ceiling on one core and from 71 % to 78-81 % on 16 threads, while computing 14.5 % more cells. On Rayleigh-Taylor, with long intervals, it is neutral.
- `tl::segment_space` declares its gaps dead and accessible: allocated in every operand, readable, and undefined in written operands after the kernel. Bridging needs that declaration, all operands sharing the same offsets, and the same body (tag) on both sides of the gap.
- Gap cells are ghost cells. A tensorlib distribution already marks ghosts stale after a write ([ADR 0004](../../../docs/adr/0004-automatic-ghost-updates-in-the-graph.md)), and samurai refreshes its ghosts before reading them, so nothing has to change on the samurai side. An in-place update such as `u += dt*k` stays correct, since it only writes stale ghosts.
- The segment table is a struct of arrays: one length and one base per segment, the base shared by every operand. It is 2 to 5 % faster than one row per segment and smaller.
- The tail stays LLVM's own loop (vector body plus scalar remainder). After bridging, an overlapping last vector or a padded last vector are within 5 % of it, and `llvm.masked` is 3 to 4 times slower on NEON. Masked tails are worth another look on SVE and AVX-512 only.
- Work is chunked by cell count, never by a fixed number of segments, and never one segment per chunk. A fixed count starves threads once bridging has cut the number of segments; one segment per iteration is up to 40 % slower.
- Segments keep storage order: sorting by length is 2 to 5 times slower once the data leaves the cache. A per-cell index list is not an alternative (54-67 % of the span ceiling).

The remaining gap is samurai's layout: a bridged kernel reaches 72-80 % of a flat loop over the span, so about 50 % of a flat loop over as many cells. Closing it would mean storing ghosts apart from the cells in samurai, which is outside this effort.

Amends [Kernel IR](22-kernel-ir.md) (segmented iteration space) and [Reconcile the core with samurai and ponio](26-reconcile-with-samurai-ponio.md) (`tl::segment_space`). Not measured: 2D stencils that read neighbouring rows through other intervals, kernels with several bodies, x86 masks, GPUs; the GPU and x86 checks go to [Rerun the fusion benchmark on an HPC node and a GPU](19-rerun-fusion-benchmark-hpc-gpu.md). The prototype stays in `.scratch` as the primary source (the repository has no git history yet). The samurai meshes come from samurai `9e77d1a` with a 12-line dump helper described in the README.

### Review log

- The user chose bridging declared by `tl::segment_space` over padding only and over asking samurai to store ghosts apart.
