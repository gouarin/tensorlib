# Prototype of compile times for lighter templates

Type: prototype
Status: resolved
Blocked by: 02

## Question

Build a rough prototype of a complex expression with the lighter-template techniques found in research, and measure its compile time and run time against the same expression in xtensor.

## Answer

Decided with the user on 2026-10-30. Prototype: [prototypes/11-compile-times/](../prototypes/11-compile-times/README.md). It covers four variants of the same generated expressions: a graph-recording front-end, minimal C++20 expression templates, xtensor with stored intermediates, and xtensor lazy. All four were compiled with GCC 15.2 at `-O3` on an aarch64 VM.

- **Compile time.** With K chained, broadcasting statements in one TU, the recording front-end compiles in 0.01 s (K = 1) to 0.44 s (K = 64), with under 100 MB of memory. xtensor takes 1.1 s for a single statement, then 37 s at K = 64 with stored intermediates, and 127 s with 7.4 GB when they stay lazy, which is superlinear. Minimal expression templates take 0.07 to 1.1 s.
- **Run time (K = 16, n = 2048).** The recording front-end, even with its interpreter and no JIT, beats xtensor with stored intermediates on one thread (879 against 1 050 ms). With 16 threads it takes 81 ms against 1 020 ms, because xtensor does not parallelise broadcasting assignments by default. The expression templates recompute lazy chains for each output (3 744 ms on one thread).
- **Decision: a recording front-end.** Expressions on `tensor<T, Rank>` go through a thin front-end. Operators take a type-erased `tl::expr<T>` and record nodes through functions compiled once. The expression tree never becomes a type. Templates carry only the element type and the rank. There are no expression templates for run-time-sized tensors. This settles the template side of [Code-generation strategy](12-code-generation-strategy.md).
- **Decision: `fixed<T, N...>` is evaluated eagerly.** Its operations are direct and `constexpr`, with no expression templates: `a + b` returns a computed `fixed`. They can be called inside Kokkos kernels and generated kernels. Inside the graph, a `fixed` element type is lowered as N lanes in the [Kernel IR](22-kernel-ir.md).
- Not measured: Clang, MSVC and nvcc; views and reductions; the cost of compiling a new kernel with the JIT at run time. The generator and `measure.py` can seed the compile-time benchmarks in CI (fog "Test and benchmark infrastructure").
