# fixed<T, N...> and element traits

Status: resolved
Blocked by: 01

## Context

Spec: [Tensors, views and expressions](../../core-architecture/spec.md#3-tensors-views-and-expressions), [Extension points](../../core-architecture/spec.md#12-extension-points), [Style rules](../../core-architecture/spec.md#15-style-rules).
Decisions: [Tensor data model](../../core-architecture/issues/07-tensor-data-model.md), [Prototype of compile times for lighter templates](../../core-architecture/issues/11-compile-time-prototype.md), [Coding style and extension points](../../core-architecture/issues/29-coding-style-and-extension-points.md).
ADRs: [ADR 0011](../../../docs/adr/0011-thin-templates-over-an-erased-core-and-layered-extension-points.md).
Builds on: [Build skeleton and repository layout](01-build-skeleton.md).

## What to build

- `tl::fixed<T, N...>`: inline storage, value semantics, arithmetic and comparisons, `constexpr` throughout, evaluated eagerly (no expression templates).
- `tl::element_traits` and the `tl::element` concept for scalar types and `fixed` over them; user specialisation documented with an example.
- Only whitelisted standard headers in public headers (`<cmath>` is allowed for `fixed`).

## Acceptance

- Arithmetic, broadcasting of a scalar over components and comparisons are tested with `static_assert`.
- A user type specialising `tl::element_traits` satisfies `tl::element` in a test.
- The header check passes.

## Answer

Done on branch `fixed-element` (2026-10-31). Decided with the user while implementing:

- `tl::element_traits` describes the representation only: a `scalar` of the closed set and the component `extents` (rank 0 to 4), C order. `tl::element` also checks that the type is trivially copyable with the size and alignment of `scalar[extents...]`. Arithmetic on user element types will go through traced functions.
- `==` and `!=` on `fixed` compare whole values and return `bool`; there is no `<`. Element-wise comparisons are `tl::equal`, `tl::not_equal`, `tl::less`, `tl::less_equal`, `tl::greater` and `tl::greater_equal`, which return `fixed<bool, N...>`.
- `fixed<T, N...>` is a pure aggregate over a flat `std::array` in C order (`tl::fixed<double, 2, 2>{1., 2., 3., 4.}`), with `tl::fixed_from({{1., 2.}, {3., 4.}})` for nested braces at ranks 2 to 4.
- As many indices as the rank give an element (`T&`). Fewer give a `tl::fixed_view<T, N...>` of the block, as on tensors: assigning to it writes into the `fixed` (from a `fixed`, a view or a scalar), copying it gives another view, and converting it to `fixed` copies the values. A view of a const `fixed` is read-only, and indexing a temporary with fewer indices does not compile, so no view dangles.

Also in the branch:

- `tl::scalar`: `bool`, signed and unsigned integers of 8 to 64 bits, `float`, `double`. Complex, float16 and bfloat16 come with [Shapes, dtypes and Array API type promotion](04-shapes-dtypes-and-promotion.md).
- Arithmetic `+ - * /`, unary `-` and compound assignment between `fixed` and views of the same shape and element type, or with a scalar, which adopts the element type (Array API weak scalars). Booleans do no arithmetic. Different shapes or element types do not compile; promotion between element types also waits for the dtypes ticket.
- Host bounds checks under `TL_ENABLE_ASSERTIONS` through `tl::detail::check_index`, which aborts with the index and the extent (`src/core/assert.cpp`).
- Glossary: Fixed view and Element type.
- Examples `examples/fixed.cpp` and `examples/element_traits.cpp`; tests `tests/core/fixed.cpp` and `tests/core/element.cpp`, almost all `static_assert`.

Checked: GCC 15 and Clang 21 with warnings as errors, ASan and UBSan, clang-tidy, the header checks, and the floors GCC 11.1 and Clang 16 in their containers.

Not in this ticket: calling `fixed` from Kokkos or CUDA kernels. Its functions are `constexpr`, so `--expt-relaxed-constexpr` makes them callable on the device, but the bounds check under `TL_ENABLE_ASSERTIONS` calls a host function. [Kokkos interop kernels](28-kokkos-interop-kernels.md) settles device annotations.
