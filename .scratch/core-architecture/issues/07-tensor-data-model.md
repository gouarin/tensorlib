# Tensor data model

Type: grilling
Status: resolved
Blocked by: 16

## Question

What is a tensor in the core? Decide how the three container kinds relate, owning tensors versus views, layouts (row major, column major, strided), memory spaces, supported element types, and how a tensor relates to a partitioned tensor and its local parts.

## Answer

Decided with the user on 2026-10-30. Vocabulary is in `CONTEXT.md`. The copy semantics are recorded in [ADR 0001](../../../docs/adr/0001-value-semantics-with-copy-on-write.md).

- **One tensor type with a distribution.** Every tensor has a distribution; the default is local. A partitioned tensor is not a separate type, so the same code runs on local and partitioned data.
- **Mixing distributions.** Combining tensors with incompatible distributions is an error, and the user calls `redistribute()` explicitly. The one exception is a replicated operand, which combines with any partitioned tensor on the same communicator (for example `a(N,M) + b(M)`). A local tensor combined with a partitioned one is an error.
- **Local part.** It is a view, as a local tensor, on the owned elements. A distribution can declare a ghost width. Storage then includes the ghosts, and a separate accessor exposes owned plus ghosts (DMDA local vectors, Tpetra overlap maps). How ghosts are exchanged belongs to the "distribution hooks" fog.
- **Copy semantics.** Copies have value semantics, with copy-on-write over reference-counted storage. A view refers to its owning tensor, not the raw buffer, so writing through a view detaches the owner from any copies first. After `v = view(a); c = a; v(0) = 7;`, `a(0) == 7` and `c(0)` keeps its old value.
- **Container kinds.**
  - `tensor<T, Rank = dynamic>` is one template for fixed rank and dynamic rank. Fixed to dynamic is an implicit conversion without copy. Dynamic to fixed goes through `rank_cast<R>`, explicit and checked at run time. Mixing ranks gives a dynamic-rank result.
  - `fixed<T, N...>` is a separate type for small tensors with inline storage. It is used in local computations and as the element type of larger tensors (`tensor<vec3>`), takes the memory space of its container, and is never partitioned.
- **Element types.** The element type is static in the C++ API (`tensor<double>`). A type-erased handle (`any_tensor`, with dtype, rank, memory space and layout known only at run time) serves bindings and the JIT graph, and converting between the two is explicit and checked. v1 supports f16, bf16, f32, f64, c64, c128, signed and unsigned integers from 8 to 64 bits, bool, and fixed-shape tensors of these.
- **Memory space.** A run-time attribute, not part of the type. Data moves explicitly with `to(space)`. Mixing memory spaces in one operation is an error, reported when the graph node is built.
- **Layout.** A run-time attribute: row major (C) by default, column major (Fortran) and strided also allowed. Logical semantics (`reshape`, `ravel`) always follow NumPy C order, whatever the physical layout. Generated kernels choose the iteration order.

## Amendment (2026-10-30)

[Reconcile the core with samurai and ponio](26-reconcile-with-samurai-ponio.md): the ghost width applies to regular distributions only. Codes such as samurai, whose ghosts sit among owned cells in their own numbering, use local tensors and keep their own communication.
