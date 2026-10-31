# Broadcasting design

Type: grilling
Status: resolved
Blocked by: 05, 07, 16

## Question

How does the core implement broadcasting so that it fuses with other operations, stays correct for the edge cases found in research, and extends to partitioned tensors?

## Answer

Decided with the user on 2026-10-30, from [Broadcasting semantics and edge cases](05-broadcasting-semantics.md). The vocabulary is in `CONTEXT.md`.

- **Rule and checks.** tensorlib follows the Array API algorithm exactly. The tests are "is 1", not a maximum, so `1` against `0` gives `0` and `0` against `n > 1` is an error. Shapes are computed and checked when the graph node is built, so a mismatch is raised at the line that caused it, even though evaluation is deferred.
- **In-place.** In-place and compound assignment never change the target's shape. The right-hand side must broadcast to the target's shape, otherwise it is an error. This differs from xtensor, which resizes the target.
- **Explicit broadcasting.** `tl::broadcast_to` and `tl::broadcast_arrays` return expressions (`tl::expr`). Expressions are read-only and never stored, which rules out writing through a broadcast. Exporting one (DLPack, mdspan, Kokkos) means first assigning it to a `tl::tensor`. `expand_dims` and new size-1 axes duplicate nothing, so they stay writable views, as in NumPy.
- **One affine index map per leaf.** Inside a fused kernel, every leaf reads its buffer through one affine index map: an offset plus one stride per result dimension. A stretched dimension has stride 0, and a dimension the leaf lacks has no entry. Strided slices, transposes and stencil shifts compose into the same map. Stride 0 exists only in the kernel IR, never in a Kokkos View or an mdspan. Aliasing analysis compares maps: an output whose buffer is also read through a different map gets a copy or a kernel split, as decided in [When the task graph is evaluated](08-evaluation-trigger.md). When all shapes are equal, the kernel is a flat loop.
- **Fixed-shape elements.**
  - `tensor<vec3> + vec3` broadcasts the fixed value to every element.
  - A `tensor<double>` of the same shape scales every component of a `tensor<vec3>`.
  - Anything that mixes tensor axes with components goes through the `as_components(u)` view, which exposes the components as a last axis (shape `{N, M, 3}`), and then follows the plain rule.
  - `u + tensor<double>({3})` is an ambiguity error.
- **Partitioned tensors.** Broadcasting is defined on the global shape. A partitioned operand is compatible only if its partition matches the result's on every dimension it really has, and if it is not partitioned along any stretched dimension (for example `a(N,M)` split by rows plus `b(N,1)` split by rows). A replicated operand is always compatible. Any case that would need communication is an error that suggests `replicate()` or `redistribute()`. A result is never written through a broadcast.
- **Left to [Kernel IR](22-kernel-ir.md):** how the IR encodes the map, and whether kernels are specialised and cached per broadcast pattern (which dimensions are stretched) or per concrete extents.

## Amendment (2026-10-30)

[Reconcile the core with samurai and ponio](26-reconcile-with-samurai-ponio.md): the affine index map holds per segment of a segmented iteration space, not per operand. Each operand has one base offset per segment (samurai intervals) and an affine map inside it. A plain tensor is the one-segment case.
