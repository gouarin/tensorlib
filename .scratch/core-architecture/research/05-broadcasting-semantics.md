# Broadcasting semantics and edge cases

Research for ticket `issues/05-broadcasting-semantics.md`.

Sources are read at these commits (October 2026), cloned from GitHub. numpy.org and data-apis.org were blocked by the sandbox firewall, so the docs are cited from their source files in the same repos (these are the files the published pages are built from).

| Project | Repo | Commit |
|---|---|---|
| NumPy | numpy/numpy | `5cc98a3f` |
| Array API standard | data-apis/array-api | `ff497ed8` (draft spec) |
| xtensor | xtensor-stack/xtensor | `57f3d07e` |
| mdspan reference impl | kokkos/mdspan | `a32a23b5` |
| Kokkos | kokkos/kokkos | `d38f4d5b` |
| cuPyNumeric | nv-legate/cupynumeric | `4e0e53ef` |
| Legate | nv-legate/legate | `880b2756` |

## 1. The rule

NumPy and the Array API define the same rule. Align shapes on the right. Treat missing leading dimensions as 1. For each dimension pair `(d1, d2)`: if `d1 == 1` the result is `d2`, else if `d2 == 1` the result is `d1`, else if `d1 == d2` the result is `d1`, else raise an error. The result rank is the maximum rank.

- Array API algorithm, step by step: [broadcasting.rst](https://github.com/data-apis/array-api/blob/ff497ed83220372eb689a4fb0878c831470f93bc/spec/draft/API_specification/broadcasting.rst)
- NumPy user guide, "General broadcasting rules": [basics.broadcasting.rst](https://github.com/numpy/numpy/blob/5cc98a3f343b6efb0b731ee23d5d14fa50a2962d/doc/source/user/basics.broadcasting.rst)
- NumPy implementation, in `npyiter_fill_axisdata`: `if (bshape == 1) broadcast_shape = op_shape; else if (bshape != op_shape && op_shape != 1) goto broadcast_error;` ([nditer_constr.c L1520-L1529](https://github.com/numpy/numpy/blob/5cc98a3f343b6efb0b731ee23d5d14fa50a2962d/numpy/_core/src/multiarray/nditer_constr.c#L1520-L1529))

Only leading singleton dimensions are added. `(15,3,5)` and `(15,3)` do not broadcast (Array API examples). A 0-d array (scalar) broadcasts with anything.

## 2. Edge cases

### 2.1 Zero-size dimensions

The rule is written with "is 1" tests, not with a maximum. So:

- `1` vs `0` gives `0`.
- `0` vs `n` with `n > 1` is an error.
- `0` vs `0` gives `0`.

NumPy's user guide says each result dimension is "the largest size of the corresponding dimension among the input arrays". That sentence is wrong for 0 (it would give 1 for `1` vs `0`). The Array API algorithm and NumPy's C code both give 0. A tensorlib implementation must follow the algorithm, not the "max" shortcut.

xtensor gets the run-time case right: `xt::broadcast_shape` assigns the input extent when the accumulated extent is 1 ([xstrides.hpp L742-L783](https://github.com/xtensor-stack/xtensor/blob/57f3d07eeccfff0436288a41b9994fd09330e14b/include/xtensor/core/xstrides.hpp#L742-L783)). Its compile-time path for `fixed_shape` does not. `broadcast_fixed_shape_cmp_impl` uses 0 both as the "dimension absent" sentinel and as a real extent, takes `ordinate = max(I_v, J_v)`, and its `static_assert(!I_v || ...)` skips the check when an extent is 0 ([xshape.hpp L313-L329](https://github.com/xtensor-stack/xtensor/blob/57f3d07eeccfff0436288a41b9994fd09330e14b/include/xtensor/core/xshape.hpp#L313-L329)). From reading the code (not tested here), `fixed_shape<0>` with `fixed_shape<1>` gives `fixed_shape<1>`, and `fixed_shape<0>` with `fixed_shape<5>` compiles and gives `fixed_shape<5>`. tensorlib's compile-time shape algebra must keep "absent" and "zero" apart, for example by using `std::optional`-like encodings or by padding with 1 explicitly.

What a fused kernel has to do with size 0:

- An empty iteration space must not launch a kernel (or must launch a no-op one). Kokkos range policies accept empty ranges, but the fused code generator must not divide by an extent or index element 0 for "first element" tricks.
- A reduction over an empty broadcast result returns the identity (for example `sum` of shape `(0,)` is 0).
- Shape inference must still check compatibility when the result is empty: `(0,)` with `(3,)` is an error even though nothing is computed.
- For partitioned tensors, some local parts can be empty even when the global shape is not. The kernel path must handle a zero-size local part on any rank.

### 2.2 Stride-0 views

The usual strided implementation of broadcasting gives the broadcast operand a stride of 0 along each stretched dimension.

- NumPy's iterator does this internally, and `np.broadcast_to` returns such a view, documented as "A readonly view on the original array ... more than one element of a broadcasted array may refer to a single memory location" ([_stride_tricks_impl.py L473-L515](https://github.com/numpy/numpy/blob/5cc98a3f343b6efb0b731ee23d5d14fa50a2962d/numpy/lib/_stride_tricks_impl.py#L473-L515)). `np.broadcast_arrays` returns writeable views, but writing to them is deprecated since 1.17 and is meant to become an error ([L595-L614](https://github.com/numpy/numpy/blob/5cc98a3f343b6efb0b731ee23d5d14fa50a2962d/numpy/lib/_stride_tricks_impl.py#L595-L614)).
- xtensor goes further: every container stores stride 0 for any dimension of extent 1, so broadcasting needs no special case in its steppers ([xstrides.hpp L505-L534](https://github.com/xtensor-stack/xtensor/blob/57f3d07eeccfff0436288a41b9994fd09330e14b/include/xtensor/core/xstrides.hpp#L505-L534); documented in [numpy-differences.rst](https://github.com/xtensor-stack/xtensor/blob/57f3d07eeccfff0436288a41b9994fd09330e14b/docs/source/numpy-differences.rst)). The cost is that `a.strides()` differs from NumPy's and needs `xt::strides(a, xt::stride_type::normal)` to recover storage strides.
- `std::mdspan`'s `layout_stride` cannot represent a stride-0 view. Its mapping constructors have the precondition `s[i] > 0` for all `i`, and `is_always_unique()` is `true` (reference implementation [layout_stride.hpp L369-L370, L570-L581](https://github.com/kokkos/mdspan/blob/a32a23b523953f9c4ce0efccd56b7563bf6d83db/include/mdspan/impl/layout_stride.hpp#L369)). The layout-mapping requirements do allow a user-defined mapping whose `is_unique()` is `false`, so a broadcast view in mdspan terms needs a custom layout.
- Kokkos has no broadcasting. In current Kokkos, `View` is built on mdspan (`Kokkos_BasicView.hpp`) and `LayoutStride` maps to mdspan `layout_stride` ([Kokkos_MDSpan_Layout.hpp L67-L70](https://github.com/kokkos/kokkos/blob/d38f4d5b009d5f62628e3eb0e5010fcf4190ef84/core/src/View/MDSpan/Kokkos_MDSpan_Layout.hpp#L67-L70)), so a stride-0 Kokkos view breaks the same precondition. `Kokkos::deep_copy` between two views requires equal extents and only broadcasts a scalar fill ([Kokkos_CopyViews.hpp L973, L1197-L1230](https://github.com/kokkos/kokkos/blob/d38f4d5b009d5f62628e3eb0e5010fcf4190ef84/core/src/Kokkos_CopyViews.hpp#L1197)).

Consequence for tensorlib: broadcasting should live in tensorlib's own index mapping, not in Kokkos views. Inside a fused kernel the loop runs over the result shape, and each leaf maps a result index to its own offset by dropping the leading dimensions it lacks and pinning stretched dimensions to 0. Whether that is coded as "stride 0" internally (xtensor style) is an implementation detail, but a user-visible broadcast view must be read-only, and exporting it as a Kokkos `View` or a `std::mdspan` needs either materialization or a custom non-unique layout.

xtensor also shows a cheap specialization worth keeping: `broadcast_shape` returns a `trivial_broadcast` flag (all shapes equal), and the assigner uses a flat, SIMD, linear loop only when that flag is set and strides match; otherwise it falls back to the multi-index stepper ([xassign.hpp L405-L470](https://github.com/xtensor-stack/xtensor/blob/57f3d07eeccfff0436288a41b9994fd09330e14b/include/xtensor/core/xassign.hpp#L405-L470)). A fused generator can make the same split at code-generation time when shapes are known.

### 2.3 Writing through a broadcast and in-place assignment

The Array API sets the rule for in-place operations, including `__setitem__`: they "must not change the shape of the in-place array as a result of broadcasting", and are only supported when the right-hand side broadcasts to the left-hand side's shape after indexing ([broadcasting.rst, "In-place Semantics"](https://github.com/data-apis/array-api/blob/ff497ed83220372eb689a4fb0878c831470f93bc/spec/draft/API_specification/broadcasting.rst)). So `x[...] = a` with `x: (2,3,4)` and `a: (1,3,4)` is fine; `x[1, ...] = a` is not.

NumPy enforces the same thing in its iterator: an output operand whose shape differs from the broadcast shape raises "non-broadcastable output operand with shape ... doesn't match the broadcast shape ..." ([nditer_constr.c L1840-L1870](https://github.com/numpy/numpy/blob/5cc98a3f343b6efb0b731ee23d5d14fa50a2962d/numpy/_core/src/multiarray/nditer_constr.c#L1840)). An output that would need a stride-0 dimension is treated as a reduction and requires `NPY_ITER_REDUCE_OK` ("output operand requires a reduction", [L1400-L1410](https://github.com/numpy/numpy/blob/5cc98a3f343b6efb0b731ee23d5d14fa50a2962d/numpy/_core/src/multiarray/nditer_constr.c#L1400)).

xtensor departs from this. Its computed assignment (`a += b`) computes the broadcast shape and, if it is larger than `a`'s, evaluates into a temporary of the larger shape and moves it into `a`, which resizes `a` ([xassign.hpp L489-L523](https://github.com/xtensor-stack/xtensor/blob/57f3d07eeccfff0436288a41b9994fd09330e14b/include/xtensor/core/xassign.hpp#L489-L523)). Plain assignment `b = a + b` also resizes `b`, and xtensor uses a temporary by default for exactly that aliasing reason; `xt::noalias` turns the temporary off ([container.rst, "Aliasing and temporaries"](https://github.com/xtensor-stack/xtensor/blob/57f3d07eeccfff0436288a41b9994fd09330e14b/docs/source/container.rst)). tensorlib should follow the Array API: compound assignment never changes the target's shape, which also keeps fixed-shape and fixed-rank targets well-typed.

### 2.4 Output aliasing a broadcast input

Three cases, with how the references handle them:

1. Output and input are the same array with the same strides (`a = a + 1`, `a += b`). Element-wise evaluation in iteration order is safe. NumPy skips the copy in that case, but only if the array has no internal overlap ([nditer_constr.c L3105-L3190](https://github.com/numpy/numpy/blob/5cc98a3f343b6efb0b731ee23d5d14fa50a2962d/numpy/_core/src/multiarray/nditer_constr.c#L3105-L3190)).
2. Output partially overlaps an input (`a[1:] = a[:-1] + 1`, shifted stencil views), or the input is a broadcast view of the output's own memory (`a[:] = a[0] * 2`, where `a[0]` is broadcast along axis 0 and read after row 0 has been overwritten). NumPy's ufuncs pass `NPY_ITER_COPY_IF_OVERLAP`; the iterator runs `solve_may_share_memory` and makes the output a temporary with write-back when any read operand may share memory with it ([ufunc_object.c L1127](https://github.com/numpy/numpy/blob/5cc98a3f343b6efb0b731ee23d5d14fa50a2962d/numpy/_core/src/umath/ufunc_object.c#L1127), nditer_constr.c as above). cuPyNumeric does the same at the task level: `binary_op` calls `_copy_if_partially_overlapping` on each input before broadcasting it, which copies unless the input and output have equal storage ([deferred.py L337-L349, L5158-L5186](https://github.com/nv-legate/cupynumeric/blob/4e0e53ef3aaffabd9b36be4c7992dc8ac7ef0eb3/cupynumeric/_thunk/deferred.py#L337)).
3. Output itself is a broadcast (stride-0) view. This is a race in parallel code, since several iterations write one location. NumPy treats it as a reduction and refuses it for plain ufunc outputs; Legate forbids partitioning a broadcast dimension of an output store (section 2.6).

For a lazy fused graph this becomes a hazard analysis at graph-build or evaluation time: for each fused kernel, any written buffer that is also read through a different index map (shifted, transposed, broadcast, or sliced differently) needs either a temporary for the output, a copy of the input, or a split into two kernels. Equal index maps need nothing. A conservative "same buffer, different map, so copy" rule is what NumPy does with `max_work = 1`, and it is a correct starting point.

### 2.5 Fixed-shape, fixed-rank and dynamic-rank operands together

The rule only needs extents and rank, so it can run partly at compile time:

- Result rank is known at compile time if and only if all operands have a static rank: it is the maximum. One dynamic-rank operand makes the result dynamic-rank.
- A result extent is static only if all operands that have that dimension have a static extent there; the value follows section 1. With a static `n > 1` and a dynamic extent, the result extent is statically `n` and a run-time check confirms that the dynamic extent is `n` or `1`. A static `1` with a dynamic extent gives a dynamic extent.
- Static mismatches (`fixed<3>` with `fixed<4>`) should be compile-time errors; anything touching a dynamic extent becomes a run-time check at expression construction.

xtensor implements a coarse version of this in `promote_shape` ([xshape.hpp L389-L520](https://github.com/xtensor-stack/xtensor/blob/57f3d07eeccfff0436288a41b9994fd09330e14b/include/xtensor/core/xshape.hpp#L389-L520)): if every operand is `fixed_shape` (or a scalar), the result is the compile-time broadcast `fixed_shape`; if every operand has a `std::array` shape (fixed rank), the result is a `std::array` of the maximum size; otherwise it is a dynamic `svector`. Mixing one `fixed_shape` with one `std::array` shape drops all static extents, and the zero-extent bug from 2.1 applies to the fixed path. `std::extents` mixes static and `dynamic_extent` per dimension, which is a finer model for tensorlib's shape type than xtensor's three-way split.

The Array API also allows a dimension to be `None` (unknown) for lazy libraries when the shape depends on data ([array_object.py `shape`](https://github.com/data-apis/array-api/blob/ff497ed83220372eb689a4fb0878c831470f93bc/src/array_api_stubs/_draft/array_object.py), [data_dependent_output_shapes.rst](https://github.com/data-apis/array-api/blob/ff497ed83220372eb689a4fb0878c831470f93bc/spec/draft/design_topics/data_dependent_output_shapes.rst)). For all non data-dependent operations tensorlib can compute shapes eagerly when an expression node is built, even if evaluation is lazy, so broadcasting errors are raised at the line that caused them. This matters for the "error reporting with lazy evaluation" open question in the map.

### 2.6 Partitioned tensors

Legate (under cuPyNumeric) is the reference design here. cuPyNumeric broadcasts each input to the output shape with store transforms before launching a task, then asks the partitioner to align the output with each input ([deferred.py `_broadcast` L1945-L1960, `binary_op` L5158-L5186](https://github.com/nv-legate/cupynumeric/blob/4e0e53ef3aaffabd9b36be4c7992dc8ac7ef0eb3/cupynumeric/_thunk/deferred.py#L1945)):

- `LogicalStore::promote(extra_dim, size)` adds a dimension and `LogicalStore::broadcast(dim, size)` stretches a size-1 dimension. Both are views; "the broadcasted values are not materialized in the physical allocation" ([logical_store.h L202-L291](https://github.com/nv-legate/legate/blob/880b2756f70a104082cbf5c12125d740597f9408/src/cpp/legate/data/logical_store.h#L202-L291)).
- When the partitioner maps a tiling of the result back to the input, the promote transform removes the fake dimension from the tile color and point ([promote.cc `invert_*`](https://github.com/nv-legate/legate/blob/880b2756f70a104082cbf5c12125d740597f9408/src/cpp/legate/data/detail/transform/promote.cc#L113-L150)). Each result tile reads the input sub-block that matches its other coordinates, so the input is replicated along the broadcast dimension: every process in a row of the process grid receives the same piece.
- The fake dimension gets a partitioning restriction: `AVOID` when the store is an input, `FORBID` when it is an output ([promote.cc L70-L83](https://github.com/nv-legate/legate/blob/880b2756f70a104082cbf5c12125d740597f9408/src/cpp/legate/data/detail/transform/promote.cc#L70-L83), [dim_broadcast.cc L61-L70](https://github.com/nv-legate/legate/blob/880b2756f70a104082cbf5c12125d740597f9408/src/cpp/legate/data/detail/transform/dim_broadcast.cc#L61-L70), restriction enum in [restriction.h](https://github.com/nv-legate/legate/blob/880b2756f70a104082cbf5c12125d740597f9408/src/cpp/legate/partitioning/detail/restriction.h#L28-L32), applied with `is_output` in [logical_store.cc L540-L542](https://github.com/nv-legate/legate/blob/880b2756f70a104082cbf5c12125d740597f9408/src/cpp/legate/data/detail/logical_store.cc#L540)). This is how Legate avoids the write race of section 2.4, case 3, across processes.

What this means for tensorlib's partitioned tensors:

- Broadcasting is defined on the global shape. Local parts never broadcast against each other directly.
- The result's partition drives the computation (owner computes). For a broadcast operand, each local part of the result needs the slice of the operand obtained by projecting its index range onto the operand's dimensions and pinning stretched dimensions to 0. If the operand is partitioned along a dimension that the result also partitions the same way, no data moves. Otherwise this is a gather or a replication, and it is the same communication hook as halo exchange, so it belongs in the "distribution hooks in the graph" open question.
- A size-1 or missing dimension should not be partitioned in a broadcast operand; it is replicated. Preferring partitions that do not split broadcast dimensions (Legate's `AVOID`) keeps the replication cheap (for example `a(N,M) + b(M)` with `a` split by rows: every rank holds all of `b`, which is small).
- A broadcast view must never be the output of a partitioned write.
- The reference workload "reductions with broadcasting" (for example `a - a.mean(axis=0)`) combines a cross-process reduction with a replicated broadcast of its result; that pattern is worth a prototype once the graph exists.

## 3. Summary of requirements for tensorlib

1. Implement the Array API algorithm exactly (tests with "is 1", not max): `1` vs `0` gives `0`; `0` vs `n>1` is an error.
2. Compute shapes eagerly at expression-node construction, with compile-time evaluation wherever ranks and extents are static, and keep "absent dimension" distinct from "extent 0" in compile-time shape encodings.
3. Broadcasting is an index-map property of a leaf inside a fused kernel, implemented by tensorlib, not by Kokkos views or `layout_stride`. User-visible broadcast views are read-only; exporting them needs materialization or a custom non-unique mdspan layout.
4. Keep a fast path for the trivial case (all shapes equal, compatible strides) that runs as a flat, vectorizable loop.
5. In-place and compound assignment never change the target's shape; the right-hand side must broadcast to it.
6. Detect aliasing per fused kernel: equal index maps are safe; any other read of a written buffer (shifted, broadcast, sliced differently) forces a copy, a temporary output, or a kernel split.
7. For partitioned tensors, broadcast on the global shape; replicate broadcast operands along stretched dimensions, avoid partitioning those dimensions, and forbid writing through a broadcast.
