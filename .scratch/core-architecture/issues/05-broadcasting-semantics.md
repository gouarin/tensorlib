# Broadcasting semantics and edge cases

Type: research
Status: resolved

## Question

What exactly are the NumPy and Array API broadcasting rules, and which edge cases must a fused, lazy, possibly partitioned implementation handle? Cover zero-size dimensions, stride-0 views, in-place assignment where the output aliases a broadcast input, broadcasting between fixed-shape and dynamic-rank tensors, broadcasting of partitioned tensors, and how xtensor, Kokkos, mdspan and Legate implement it.

## Answer

NumPy and the Array API share one rule, applied right to left with "is 1" tests, not a max: `1` vs `0` gives `0`, and `0` vs `n>1` is an error. xtensor's compile-time `fixed_shape` path gets this wrong.
Implement broadcasting as an index map on each leaf inside tensorlib's fused kernels. Do not rely on stride-0 Kokkos views or `std::layout_stride`, which both require strides > 0. Broadcast views are read-only. Keep a flat fast path for the trivial case where all shapes are equal.
Follow the Array API for in-place assignment: the target's shape never changes (xtensor resizes it). When a written buffer is read through a different index map (shifted, broadcast, sliced), copy or split the kernel. NumPy and cuPyNumeric both copy on overlap.
Compute shapes eagerly at node construction, statically where the extents are static. Partitioned tensors broadcast on the global shape and replicate the broadcast operand along stretched dimensions. Like Legate, avoid partitioning those dimensions on inputs and forbid it on outputs.

Findings: [research/05-broadcasting-semantics.md](../research/05-broadcasting-semantics.md)
