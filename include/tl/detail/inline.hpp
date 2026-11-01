#pragma once

// TL_ALWAYS_INLINE marks a function the compiler must inline whatever its size.
//
// The element-wise operations of tl::fixed are small once inlined, but their bodies grow
// with the number of elements before inlining, and the inlining heuristics of clang then
// leave them as calls from 32 elements on: the temporaries go through memory and the
// operators run 1.5 to 2 times slower than a hand-written loop (benchmarks/fixed.cpp).
// Forcing inlining makes them as fast as the loop at every size measured.
//
// The attribute has no portable spelling: MSVC rejects [[gnu::always_inline]] (C5030) and
// GCC rejects [[msvc::forceinline]], so this is a macro.
#if defined(_MSC_VER) && !defined(__clang__)
#define TL_ALWAYS_INLINE __forceinline
#else
#define TL_ALWAYS_INLINE [[gnu::always_inline]]
#endif
