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

- `tl::element_traits` describes the representation only: a `scalar` of the closed set and the component `extents` (any rank, 0 included), C order. The first version capped the rank at 4; the user removed the cap on 2026-11-01. `tl::element` also checks that the type is trivially copyable with the size and alignment of `scalar[extents...]`. Arithmetic on user element types will go through traced functions.
- `==` and `!=` on `fixed` compare whole values and return `bool`; there is no `<`. Element-wise comparisons are `tl::equal`, `tl::not_equal`, `tl::less`, `tl::less_equal`, `tl::greater` and `tl::greater_equal`, which return `fixed<bool, N...>`.
- `fixed<T, N...>` stores a flat `std::array` in C order. It is written with nested braces at any rank (`tl::fixed<double, 2, 2>{{1., 2.}, {3., 4.}}`) or as a flat list of every element (`tl::fixed<double, 2, 2>{1., 2., 3., 4.}`). The first version was a pure aggregate with `tl::fixed_from` for nested braces at ranks 2 to 4, because C++ cannot deduce a shape from nested braces at an arbitrary rank. On 2026-11-01 the user asked for nested braces at any rank: a private base generates two constructors whose parameter counts come from the shape (one C array per leading index, or one scalar per element), so nothing is deduced. `fixed` is no longer an aggregate but stays trivially copyable and standard-layout, with the layout of `T[N...]`; `fixed_from` is removed.
- As many indices as the rank give an element (`T&`). Fewer give a `tl::fixed_view<T, N...>` of the block, as on tensors: assigning to it writes into the `fixed` (from a `fixed`, a view or a scalar), copying it gives another view, and converting it to `fixed` copies the values. A view of a const `fixed` is read-only, and indexing a temporary with fewer indices does not compile, so no view dangles.

Also in the branch:

- `tl::scalar`: `bool`, signed and unsigned integers of 8 to 64 bits, `float`, `double`. Complex, float16 and bfloat16 come with [Shapes, dtypes and Array API type promotion](04-shapes-dtypes-and-promotion.md).
- Arithmetic `+ - * /`, unary `-` and compound assignment between `fixed` and views of the same shape and element type, or with a scalar, which adopts the element type (Array API weak scalars). Booleans do no arithmetic. Different shapes or element types do not compile; promotion between element types also waits for the dtypes ticket.
- Host bounds checks under `TL_ENABLE_ASSERTIONS` through `tl::detail::check_index`, which aborts with the index and the extent (`src/core/assert.cpp`).
- Glossary: Fixed view and Element type.
- Examples `examples/fixed.cpp` and `examples/element_traits.cpp`; tests `tests/core/fixed.cpp` and `tests/core/element.cpp`, almost all `static_assert`.

- Documentation, at the user's request, written with the code from this ticket on: Doxygen comments on every public entity of `include/tl/`, and a Sphinx site in `docs/source/` (MyST Markdown, Breathe, furo, as in samurai) with pages for `fixed`, element types and the version. The tools are pinned in `docs/environment.yml`; `sphinx-build -W -n` runs Doxygen, which fails on any undocumented public entity or parameter. A CI job builds the site and uploads it as an artifact.
- Readability of the interface after reading it in the docs: the comparisons and unary minus return `auto` (the docs give the type), and a view's scalar assignment is constrained by `std::is_arithmetic_v` instead of an internal concept.
- Two MSVC bugs, reproduced on Compiler Explorer:
  - MSVC before 19.50 (19.44 is the one on the hosted `windows-2022` runners) cannot read a nested-brace temporary bound to an array reference in constant evaluation (C2131, "out of range index"), whatever the form of the parameter. Nested braces work there at run time; the documentation of `fixed` says so, and the `constexpr` tests of nested braces are skipped on those versions while a run-time test covers every compiler.
  - MSVC 19.51 rejects the computed template argument `std::array{N...}[K + I]...` that built the type of a view; a partial specialisation that drops the first K extents replaces it, on every compiler.
- The index computation is a loop over an array of indices instead of a fold expression, which reads better.
- Performance, at the user's request (2026-11-01): `benchmarks/fixed.cpp` (option `TL_BUILD_BENCHMARKS`) compares the operators with a hand-written loop on the same array of `fixed` and with a flat array, and `ctest -L bench` checks the limits in `benchmarks/limits.toml` in CI (time ratios, and no call left in a kernel). The element-wise helpers are loops with two hints from `tl/detail/codegen.hpp`: `TL_ALWAYS_INLINE` on the helpers and operators, because Clang leaves the unrolled helpers as calls from 32 elements on, and `TL_UNROLL_ELEMENTS` (`#pragma GCC unroll 64`, GCC only), because GCC unrolls too late to keep the temporaries in registers. Unrolling by hand (a fold over an `index_sequence` up to a size in bytes) was tried first: it made Clang spill on x86-64 SSE2, where 16 registers of 16 bytes hold 256 bytes, and a threshold per target did not fit GCC and Clang at once. Result, fixed time over loop time: about 1.0 on GCC 15 and Clang 21 from 24 to 288 bytes, with two exceptions on Clang for aarch64. First, Clang does not contract into FMA across a temporary under its default `-ffp-contract=on` (vec8f 1.6, vec3 1.17; 1.0 with `-ffp-contract=fast`). Second, `out += s * a` on vec3 (1.76) is an LLVM vectoriser choice (`ld3`/`st3` after a temporary) that plain C reproduces.

Checked: GCC 15 and Clang 21 with warnings as errors, ASan and UBSan, clang-tidy, the header checks, and the floors GCC 11.1 and Clang 16 in their containers.

Not in this ticket: calling `fixed` from Kokkos or CUDA kernels. Its functions are `constexpr`, so `--expt-relaxed-constexpr` makes them callable on the device, but the bounds check under `TL_ENABLE_ASSERTIONS` calls a host function. [Kokkos interop kernels](28-kokkos-interop-kernels.md) settles device annotations.
