#pragma once

// Code generation hints for the element-wise operations of tl::fixed. Without them, the
// operators run up to several times slower than a hand-written loop over the elements
// (benchmarks/fixed.cpp, whose limits CI checks). The hints have no portable spelling,
// so they are macros.

// TL_ALWAYS_INLINE marks a function the compiler must inline whatever its size. Once
// Clang has unrolled the loop of an element-wise helper, its inlining heuristics find the
// body too large and leave it as a call from 32 elements on, so the temporaries go through
// memory. MSVC rejects [[gnu::always_inline]] (C5030) and GCC [[msvc::forceinline]].
#if defined(_MSC_VER) && !defined(__clang__)
#define TL_ALWAYS_INLINE __forceinline
#else
#define TL_ALWAYS_INLINE [[gnu::always_inline]]
#endif

// TL_UNROLL_ELEMENTS, before a loop over the elements of a fixed, makes GCC unroll it
// completely up to 64 elements. GCC unrolls such loops too late to keep the temporaries
// in registers (1.3 times slower than the hand-written loop at 32 elements, 3.6 times at
// 36). Clang does better on its own: unrolling more makes it spill on x86-64 SSE2.
#if defined(__GNUC__) && !defined(__clang__)
#define TL_UNROLL_ELEMENTS _Pragma("GCC unroll 64")
#else
#define TL_UNROLL_ELEMENTS
#endif
