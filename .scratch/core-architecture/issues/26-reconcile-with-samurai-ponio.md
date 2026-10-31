# Reconcile the core with samurai and ponio

Type: grilling
Status: resolved
Blocked by: 25

## Question

[Requirements from samurai and ponio](25-samurai-ponio-requirements.md) found conflicts between the two reference libraries and decisions already taken: samurai meshes have no regular global index space ([Tensor data model](07-tensor-data-model.md)), their ghosts sit among owned cells rather than in a band, their access is affine only inside each interval ([Broadcasting design](13-broadcasting-design.md)), most of their physics is per-element user lambdas that [ADR 0003](../../../docs/adr/0003-runtime-backend-interface-on-lowered-kernels.md) keeps out of the IR, and their windows will be small because interval loops read results often. Decide how the core changes, or which decisions stand, so that samurai and ponio get optimal performance, and update the affected tickets.

## Answer

Decided with the user on 2026-10-30, from [Requirements from samurai and ponio](25-samurai-ponio-requirements.md).

- **samurai fields are local tensors.** samurai keeps its mesh, its load balancing and its MPI exchanges. tensorlib stores the fields, runs the kernels, and provides segmented gather and scatter kernels, so halo packing stays on the device and the buffers go to GPU-aware MPI. [Distribution kinds](18-distribution-kinds.md) covers regular distributions only, and index-list distributions (Tpetra `Map`, PetscSF) are left for later. The ghost band of [Tensor data model](07-tensor-data-model.md) applies to regular distributions only; samurai's interleaved ghosts are simply part of a local tensor.
- **Segmented iteration spaces.** The kernel IR accepts an iteration space made of a list of segments (samurai intervals). Each operand has one base offset per segment and an affine map inside the segment (stride, level shift). The segment tables are kernel arguments, rebuilt by samurai at every adaptation, and never part of the kernel cache key. This amends [Broadcasting design](13-broadcasting-design.md): the affine index map holds per segment, not per operand.
- **Traced generic lambdas.** A generic lambda (`[](auto u) {...}`) written with tensorlib operations and `fixed` values is traced once with symbolic arguments through `tl::apply`. It records its operations into the IR and fuses with the rest of the window. [ADR 0003](../../../docs/adr/0003-runtime-backend-interface-on-lowered-kernels.md) stands, since no user code enters the IR, only what it records. Data-dependent branches go through `tl::where`. What cannot be traced (loops on data, atomics written by hand) goes through Kokkos interop. Because `fixed` is evaluated eagerly ([Prototype of compile times for lighter templates](11-compile-time-prototype.md)), the same generic lambda also runs on real `fixed` values.
- **Interop kernels join the window.** A Kokkos interop kernel declares what it reads and writes and enters the pending window as a non-fusible node. It no longer forces evaluation of unrelated nodes, and it does not block the host. This amends [Backend interface](10-backend-interface.md), which made it a trigger. The fixed cost of an evaluation gets a CI budget, because samurai produces small windows.
- **Requirements passed to open tickets:**
  - [Kernel IR](22-kernel-ir.md): scatter-add outputs (atomic or coloured), fused reductions, a cache keyed on structure rather than extents or run-time scalars, 64-bit signed offsets, and up to 20 operands per kernel (ponio `rkc_202`);
  - [Views and slicing](23-views-and-slicing.md): writable views built from interval tables, `uint8` tags with bitwise masks;
  - [Buffer reuse and allocation](21-buffer-reuse-and-allocation.md): cheap reallocation at a new size every step, and staggered start offsets for large buffers (ponio PR #115 lost 43 %);
  - [Zero-copy interop with PETSc and Tpetra](17-petsc-tpetra-interop.md): handing out the local buffer like `VecCreateSeqWithArray`.
- **Consequence for porting.** samurai's current host loops over `field.data()` will not survive the move to GPUs unchanged. They become tensorlib expressions over segmented views, traced lambdas, or interop kernels.
