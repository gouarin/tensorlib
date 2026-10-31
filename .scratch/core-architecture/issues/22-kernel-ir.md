# Kernel IR

Type: grilling
Status: resolved
Blocked by: 10, 12, 13, 23

## Question

What is the internal IR that the core hands to a backend for one fused kernel ([Backend interface](10-backend-interface.md))? Decide how it represents the n-d iteration space, the index maps (broadcasting, transpose, shifts for stencils, strided views; see [Broadcasting design](13-broadcasting-design.md)), element types including `fixed<T, N...>` elements, reductions and scans as nodes (axes, identities, multiple outputs), and the information per-target lowering needs from the declared capabilities. Also decide how the IR keys the kernel cache (graph signature, shapes symbolic or concrete), and how the serial interpreter and the JIT lowerings share it. The answer must fit the choice made in [Code-generation strategy](12-code-generation-strategy.md).

Requirements from [Reconcile the core with samurai and ponio](26-reconcile-with-samurai-ponio.md):

- segmented iteration spaces, with one base per operand per segment and an affine map inside, and segment tables as kernel arguments;
- operations recorded from generic lambdas traced by `tl::apply`, and `tl::where`;
- scatter-add outputs (atomic or coloured) and fused reductions;
- a cache keyed on structure rather than extents or run-time scalars;
- 64-bit signed offsets;
- up to 20 operands per kernel;
- interop kernels as window nodes.

Requirement from [Zero-copy interop with PETSc and Tpetra](17-petsc-tpetra-interop.md): a non-fusible external-operator node (PETSc `MatMult`, Tpetra `apply`) that declares its reads and writes, like interop kernels.

Requirements from [Views and slicing](23-views-and-slicing.md): indirect loads (gather by an integer tensor), scatter and accumulating scatter-add from index views, masked writes as an in-place `where`, negative strides (`flip`), segment lengths shared by every operand of a segment space with one base-offset table per operand, and the fused detach of partial writes on a shared buffer, whose regions (interior, boundaries) become segments of one kernel.

Requirements from [Error reporting and debug mode](24-error-reporting-and-debug-mode.md): every node carries its operation and the index of its `tl::site`, every kernel carries the list of sites fused into it (used for its name in Kokkos Tools and in error reports), and the lowering can insert per-node checks (`bounds`, `nan` from finite inputs, `scatter`) without changing the kernel's meaning.

Requirements from [Code-generation strategy](12-code-generation-strategy.md): a canonical, hashable form in which arguments (extents, offsets, strides, segment tables, scalars) are separate from structure; a per-operand "inner stride is 1" flag; reductions expressed over canonical blocks of fixed size; a printable text form for `TL_DEBUG=dump`; a lowering per operation to LLVM IR with a table of intrinsics per target; a kernel-splitting interface driven by the two ceilings (operations, live values).

## Answer

Decided with the user on 2026-10-30. Vocabulary (evaluation plan, kernel body, combiner) is in `CONTEXT.md`. The two levels and the shape of the IR are recorded in [ADR 0009](../../../docs/adr/0009-evaluation-plan-and-per-element-ssa-kernel-ir.md).

- **Two levels.** The evaluation plan orders kernels and non-fusible nodes that declare their reads and writes: library operations, Kokkos interop kernels, `tl::petsc::mult` and `tl::tpetra::apply`, ghost exchanges, and reductions across ranks (which carry the poison status of [ADR 0007](../../../docs/adr/0007-errors-poison-results-across-ranks.md)). The kernel IR describes one fusible kernel only. Backends compile kernels without knowing about communication or libraries.
- **Shape of a kernel.** A per-element SSA program over an iteration space, either n-d rectangular or segmented (1-d segments with trailing ordinary axes).
  - Operands carry an element type and either an affine map per segment (base offset, one stride per dimension, a component stride for `fixed` elements) or a gather whose indices are loaded from another operand. Stride 0 (broadcast), the "inner stride is 1" flag and stencil shifts are structural; other strides (negative for `flip`, 2 for samurai level shifts), offsets, extents and segment tables are arguments.
  - Arguments are run-time scalars and tables.
  - Each SSA operation carries its operation and the index of its `tl::site`; operations recorded by `tl::apply` are ordinary SSA operations. `tl::where` is a `select`.
  - Outputs are stores (with an optional mask, which is how `u(mask) = e` becomes an in-place `where`), reductions and scatters.
  - There are no loops, tiles or shared memory in the IR. Lowering decides them per target from capabilities; GPU lowering recognises stencils from the shifts to tile them.
- **Fixed elements.** A `fixed<T, N...>` value is one lane-typed SSA value (`f64x5`); `lane` extracts a component. It is loaded through the affine map plus a component stride (1 in AoS, the array length in SoA), which enters the key like other strides. AoS is the default layout; SoA can be added later as a tensor attribute without changing the IR. Lowering scalarises lanes for LLVM, and the kernel-size ceilings count lanes.
- **Scalar types.** The Array API set (`bool`, `int8` to `int64`, `uint8` to `uint64`, `float32`, `float64`, `complex64`, `complex128`), plus `float16` and `bfloat16` as storage types computed in `float32` (convert on load, round on store). The set is closed; a new scalar type is a core change versioned in the cache key. The IR has no implicit conversions: the front end applies Array API promotion and records explicit `cast` operations. User types are built by composition, `fixed` over base types and traced lambdas (a double-double is a traced lambda over `fixed<double, 2>`).
- **Reductions and scans.**
  - Built-in combiners (`sum`, `prod`, `min`, `max`, `all`, `any`, `argmin`, `argmax`) with an optimised lowering per target, and traced combiners through `tl::reduce(e, init, combine)`: an associative lambda over a `fixed` state with its identity, recorded as a small SSA region (Welford, Kahan, multi-dimensional argmin). Associativity is declared by the user and checked in debug mode by comparing two splits. Commutativity is not needed, since the canonical block order of [Code-generation strategy](12-code-generation-strategy.md) is fixed.
  - A reduction is a kernel output with its reduced axes (all or some), its combiner and its identity. One kernel can hold several stores and several reductions over the same iteration space.
  - A scan is a kernel of its own: its input expression is fused as a prologue, it has no other outputs in v1, and it uses the same combiners.
  - On a partitioned tensor the kernel produces the local result, and the plan adds the reduction across ranks.
- **Scatter-add.** Affine scatter-adds (`u(I) -= f; u(I + 1) += f;` on boxes or segments) are turned into gathers by lowering, deterministic and without atomics. Indirect scatter-adds through an index tensor use atomics by default. A deterministic path (sort by target index, then a segmented reduction in fixed order, with the sort cached per version of the `idx` buffer) is used under `TL_FP=strict` or for operands wrapped in `tl::deterministic(idx)`. Floating-point indirect scatter-add outside that path is the documented exception to the reproducibility of [Code-generation strategy](12-code-generation-strategy.md). Colouring stays a possible internal optimisation. Integer scatter-add is exact either way.
- **Several bodies.** A kernel can hold several bodies (SSA regions), and each segment of the iteration space carries the index of the body it runs, as one more column of the segment table. This covers interior and boundary schemes in one launch and the fused detach of [Views and slicing](23-views-and-slicing.md) (new values in the written regions, copies of old values elsewhere). Bodies are part of the key; segments and their tags are arguments. A plain tensor is one segment with one body.
- **Index width.** The IR is 64-bit signed in meaning, and offsets, extents and tables are stored as `int64`. A backend capability allows a 32-bit variant: at launch, the plan checks from extents and strides, in O(operands), that every address fits in `int32` (for a gather, that the target buffer fits), and selects that variant. The flag is part of the key and changes only when an array crosses 2^31 elements. The capability is off by default on CPU and on for GPUs, pending measurement.
- **Text, serialisation, hashing.** A versioned text form that prints and parses (round trip), used by `TL_DEBUG=dump` and by tests written as text files run through `tl-opt` and FileCheck. A binary serialisation for kernel bundles. A structural hash computed incrementally while recording, never by serialising on the hot path. The format version enters the key. Tests check `parse(print(ir)) == ir` and hash stability. The printer and parser live in the compiled library, not in public headers.
- **Requirements met from earlier tickets.** Segmented spaces with tables as arguments, traced lambdas and `where`, scatter-add and fused reductions, a structural key, 64-bit signed offsets, up to 20 operands, gathers, masked writes, negative strides, shared segment lengths with one offset table per operand, fused detach, sites and per-node checks, canonical reduction blocks, the inner-stride flag, the text form and per-operation lowering. Interop kernels and external operators are plan nodes, not kernel IR. The splitting interface of [Code-generation strategy](12-code-generation-strategy.md) works on the SSA program and counts operations and live lanes.
- **Follow-ups.** [Prototype of the LLVM JIT on CPU](27-llvm-jit-cpu-prototype.md) adds an affine scatter-add turned into a gather and a two-body kernel. [Rerun the fusion benchmark on an HPC node and a GPU](19-rerun-fusion-benchmark-hpc-gpu.md) adds the 32-bit index variant and atomic scatter-add.

Grilling log (2026-10-30):

- Q1 form: option A. Per-element SSA program over an iteration space; non-fusible nodes and communication live in the evaluation plan; no loops or tiles in the IR.
- Q2 fixed elements: option C. Lane-typed SSA values loaded through the affine map plus a component stride; AoS default, SoA later; ceilings count lanes.
- Q3 scalar types: option B. Array API set plus float16 and bfloat16 as storage computed in float32; closed set; explicit casts.
- Q4 reductions: option B. Built-in plus traced associative combiners with identity; several stores and reductions per kernel; scans in their own kernel with a fused prologue.
- Q5 scatter-add: option B. Affine scatters turned into gathers; indirect ones atomic by default, deterministic sort path under TL_FP=strict or tl::deterministic(idx).
- Q6 text: option B. Round-trip versioned text, binary for bundles, incremental structural hash while recording.
- Q7 regions: option B. Several bodies per kernel, selected by a tag per segment.
- Q8 index width: option C. 64-bit semantics, 32-bit variant allowed by a backend capability and chosen at launch.

## Amendment (2026-10-31)

[Coding style and extension points](29-coding-style-and-extension-points.md): the primitive set stays closed to users, with one added operation. `call` invokes a registered external function, a pure scalar function with a required host symbol and optional LLVM bitcode per target. The generator inlines the bitcode when present and calls the host symbol on CPU otherwise; a GPU target without bitcode makes plan construction raise `tl::unsupported_operation`. The bitcode hash is part of the structural key, and the serial interpreter calls the host symbol.
