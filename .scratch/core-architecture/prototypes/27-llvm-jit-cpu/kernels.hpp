// PROTOTYPE - reference kernels in the mini IR, their hand-written C++ twins, and test data.
#pragma once
#include "jit.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <random>
#include <vector>

namespace proto {

constexpr std::int64_t chunk = 2 * canonical_block; // elements per parallel_for iteration
constexpr std::int64_t seg_chunk = 256;             // segments per parallel_for iteration

// ---- kernels in the IR ----------------------------------------------------

// ponio: out = y + dt * (b1*k1 + ... + bS*kS). Operands: y, k1..kS, out. Args: dt, b1..bS.
inline kernel ponio(int stages) {
  kernel k{"ponio", stages + 2, false, {body{}}};
  body& b = k.bodies[0];
  int acc = b.bin(opk::mul, b.arg(1), b.load(1));
  for (int s = 2; s <= stages; ++s) acc = b.bin(opk::add, acc, b.bin(opk::mul, b.arg(s), b.load(s)));
  int out = b.bin(opk::add, b.load(0), b.bin(opk::mul, b.arg(0), acc));
  b.stores.push_back({stages + 1, out});
  return k;
}

// out = u - c * (u(+1) - two*u + u(-1)). Operands: u, out. Args: c, two.
inline body stencil_body() {
  body b;
  int u = b.load(0);
  int lap = b.bin(opk::add, b.bin(opk::sub, b.load(0, 1), b.bin(opk::mul, b.arg(1), u)), b.load(0, -1));
  b.stores.push_back({1, b.bin(opk::sub, u, b.bin(opk::mul, b.arg(0), lap))});
  return b;
}

// out = u * scale. Args: c, two, scale.
inline body boundary_body() {
  body b;
  b.stores.push_back({1, b.bin(opk::mul, b.load(0), b.arg(2))});
  return b;
}

inline kernel stencil_segmented() { return {"stencil", 2, true, {stencil_body()}}; }
inline kernel two_body() { return {"two_body", 2, true, {stencil_body(), boundary_body()}}; }
inline kernel boundary_only() { return {"boundary", 2, true, {boundary_body()}}; }

// Affine scatter-add `u(I) -= f; u(I + 1) += f;` turned into a gather: out = u + f - f(+1).
// Operands: u, f, out.
inline kernel flux_gather() {
  kernel k{"flux", 3, false, {body{}}};
  body& b = k.bodies[0];
  b.stores.push_back({2, b.bin(opk::sub, b.bin(opk::add, b.load(0), b.load(1)), b.load(1, 1))});
  return k;
}

// x = a; repeated `steps` times: x = x * s_(k mod 8) + b. Operands: a, b, out. 2 ops per step.
inline kernel chain(int steps) {
  kernel k{"chain" + std::to_string(steps), 3, false, {body{}}};
  body& b = k.bodies[0];
  int x = b.load(0);
  int y = b.load(1);
  for (int s = 0; s < steps; ++s) x = b.bin(opk::add, b.bin(opk::mul, x, b.arg(s % 8)), y);
  b.stores.push_back({2, x});
  return k;
}

// L inputs all live across 4 outputs: out_m = sum_j a_j * w_(m*L+j).
inline kernel live(int L) {
  kernel k{"live" + std::to_string(L), L + 4, false, {body{}}};
  body& b = k.bodies[0];
  std::vector<int> a;
  for (int j = 0; j < L; ++j) a.push_back(b.load(j));
  for (int m = 0; m < 4; ++m) {
    int acc = b.bin(opk::mul, a[0], b.arg(m * L));
    for (int j = 1; j < L; ++j) acc = b.bin(opk::add, acc, b.bin(opk::mul, a[j], b.arg(m * L + j)));
    b.stores.push_back({L + m, acc});
  }
  return k;
}

// ---- dispatch through Kokkos: Kokkos owns the threads, the JIT owns the body ------------

inline void run_flat(flat_fn f, std::int64_t n, double** bufs, const double* args) {
  std::int64_t nc = (n + chunk - 1) / chunk;
  Kokkos::parallel_for("jit_flat", Kokkos::RangePolicy<>(0, nc), [=](std::int64_t c) {
    f(c * chunk, std::min(n, (c + 1) * chunk), bufs, args);
  });
}

inline void run_seg(seg_fn f, std::int64_t nseg, double** bufs, const double* args, const std::int64_t* seg) {
  std::int64_t nc = (nseg + seg_chunk - 1) / seg_chunk;
  Kokkos::parallel_for("jit_seg", Kokkos::RangePolicy<>(0, nc), [=](std::int64_t c) {
    f(c * seg_chunk, std::min(nseg, (c + 1) * seg_chunk), bufs, args, seg);
  });
}

// Canonical reduction: blocks in parallel, then a fixed pairwise tree over block partials.
inline double run_reduce(red_fn f, const double* a, std::int64_t n, std::vector<double>& partial, bool sum) {
  std::int64_t nb = (n + canonical_block - 1) / canonical_block;
  partial.resize(std::size_t(nb));
  double* p = partial.data();
  std::int64_t per = chunk / canonical_block;
  Kokkos::parallel_for("jit_reduce", Kokkos::RangePolicy<>(0, (nb + per - 1) / per), [=](std::int64_t c) {
    f(c * per, std::min(nb, (c + 1) * per), a, n, p);
  });
  Kokkos::fence();
  for (std::int64_t w = 1; w < nb; w *= 2)
    for (std::int64_t i = 0; i + w < nb; i += 2 * w) p[i] = sum ? p[i] + p[i + w] : std::max(p[i], p[i + w]);
  return p[0];
}

// ---- hand-written twins (what a perfect generator would emit), compiled by g++ -O3 -------

inline void hand_ponio(int S, std::int64_t n, double** bufs, const double* args) {
  Kokkos::parallel_for("hand_ponio", Kokkos::RangePolicy<>(0, (n + chunk - 1) / chunk), [=](std::int64_t c) {
    std::int64_t lo = c * chunk, hi = std::min(n, lo + chunk);
    const double* __restrict y = bufs[0];
    double* __restrict out = bufs[S + 1];
    double dt = args[0];
    for (std::int64_t i = lo; i < hi; ++i) {
      double acc = args[1] * bufs[1][i];
      for (int s = 2; s <= S; ++s) acc += args[s] * bufs[s][i];
      out[i] = y[i] + dt * acc;
    }
  });
}

// Fully unrolled 20-stage loop: the loop a code generator would really emit. GCC only
// vectorises it when the restrict pointers are function parameters, not lambda locals.
[[gnu::noinline]] inline void hand_ponio20_loop(std::int64_t lo, std::int64_t hi, double* __restrict out,
                                                const double* __restrict y, const double* __restrict k1, const double* __restrict k2, const double* __restrict k3, const double* __restrict k4, const double* __restrict k5, const double* __restrict k6, const double* __restrict k7, const double* __restrict k8, const double* __restrict k9, const double* __restrict k10, const double* __restrict k11, const double* __restrict k12, const double* __restrict k13, const double* __restrict k14, const double* __restrict k15, const double* __restrict k16, const double* __restrict k17, const double* __restrict k18, const double* __restrict k19, const double* __restrict k20,
                                                const double* __restrict b) {
  for (std::int64_t i = lo; i < hi; ++i) {
      double acc = b[1] * k1[i];
      acc += b[2] * k2[i];
      acc += b[3] * k3[i];
      acc += b[4] * k4[i];
      acc += b[5] * k5[i];
      acc += b[6] * k6[i];
      acc += b[7] * k7[i];
      acc += b[8] * k8[i];
      acc += b[9] * k9[i];
      acc += b[10] * k10[i];
      acc += b[11] * k11[i];
      acc += b[12] * k12[i];
      acc += b[13] * k13[i];
      acc += b[14] * k14[i];
      acc += b[15] * k15[i];
      acc += b[16] * k16[i];
      acc += b[17] * k17[i];
      acc += b[18] * k18[i];
      acc += b[19] * k19[i];
      acc += b[20] * k20[i];
      out[i] = y[i] + b[0] * acc;
  }
}

inline void hand_ponio20(std::int64_t n, double** bufs, const double* args) {
  Kokkos::parallel_for("hand_ponio20", Kokkos::RangePolicy<>(0, (n + chunk - 1) / chunk), [=](std::int64_t c) {
    double** p = bufs;
    hand_ponio20_loop(c * chunk, std::min(n, (c + 1) * chunk), p[21], p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15], p[16], p[17], p[18], p[19], p[20], args);
  });
}

inline void hand_stencil(std::int64_t nseg, double** bufs, const double* args, const std::int64_t* seg) {
  Kokkos::parallel_for("hand_stencil", Kokkos::RangePolicy<>(0, (nseg + seg_chunk - 1) / seg_chunk), [=](std::int64_t c) {
    const double* __restrict u = bufs[0];
    double* __restrict out = bufs[1];
    double cc = args[0], two = args[1], scale = args[2];
    for (std::int64_t s = c * seg_chunk; s < std::min(nseg, (c + 1) * seg_chunk); ++s) {
      const std::int64_t* r = seg + 4 * s;
      const double* __restrict us = u + r[2];
      double* __restrict os = out + r[3];
      if (r[1] == 0)
        for (std::int64_t i = 0; i < r[0]; ++i) os[i] = us[i] - cc * ((us[i + 1] - two * us[i]) + us[i - 1]);
      else
        for (std::int64_t i = 0; i < r[0]; ++i) os[i] = us[i] * scale;
    }
  });
}

inline void hand_flux_gather(std::int64_t n, double** bufs) {
  Kokkos::parallel_for("hand_flux_gather", Kokkos::RangePolicy<>(0, (n + chunk - 1) / chunk), [=](std::int64_t c) {
    const double* __restrict u = bufs[0];
    const double* __restrict f = bufs[1];
    double* __restrict out = bufs[2];
    for (std::int64_t i = c * chunk; i < std::min(n, (c + 1) * chunk); ++i) out[i] = (u[i] + f[i]) - f[i + 1];
  });
}

// The naive lowering of `u(I) -= f; u(I + 1) += f;`: one atomic per interface side.
// f has n+1 interfaces; out starts as a copy of u. Interface i sits between cells i-1 and i.
inline void hand_flux_scatter(std::int64_t n, double** bufs) {
  Kokkos::parallel_for("hand_flux_scatter", Kokkos::RangePolicy<>(0, (n + 1 + chunk - 1) / chunk), [=](std::int64_t c) {
    const double* f = bufs[1];
    double* out = bufs[2];
    for (std::int64_t i = c * chunk; i < std::min(n + 1, (c + 1) * chunk); ++i) {
      if (i < n) Kokkos::atomic_add(&out[i], f[i]);
      if (i > 0) Kokkos::atomic_add(&out[i - 1], -f[i]);
    }
  });
}

// Free-order sum: what Kokkos gives by default (depends on the number of threads).
inline double kokkos_sum(const double* a, std::int64_t n) {
  double s = 0;
  Kokkos::parallel_reduce("kokkos_sum", Kokkos::RangePolicy<>(0, n), [=](std::int64_t i, double& acc) { acc += a[i]; }, s);
  return s;
}

inline double kokkos_max(const double* a, std::int64_t n) {
  double s = 0;
  Kokkos::parallel_reduce("kokkos_max", Kokkos::RangePolicy<>(0, n),
                          [=](std::int64_t i, double& acc) { acc = std::max(acc, a[i]); }, Kokkos::Max<double>(s));
  return s;
}

// ---- data ------------------------------------------------------------------

struct arrays {
  std::vector<std::vector<double>> store;
  std::vector<double*> ptrs;
  arrays(int count, std::int64_t n, std::uint64_t seed, std::int64_t pad = 0) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (int j = 0; j < count; ++j) {
      store.emplace_back(std::size_t(n + 2 * pad));
      for (double& x : store.back()) x = d(g);
      ptrs.push_back(store.back().data() + pad);
    }
  }
  double** data() { return ptrs.data(); }
};

// samurai-like segments: lengths 8..64, tag 1 (boundary) for every `boundary_every`-th segment.
// Row: [length, tag, base u, base out]. Segments are laid out back to back with a gap of 2 cells
// so that the stencil reads u(-1) and u(+1) inside the array.
struct segments {
  std::vector<std::int64_t> rows;
  std::int64_t count = 0, cells = 0, extent = 0;
  segments(std::int64_t target_cells, int boundary_every, std::uint64_t seed, int only_tag = -1) {
    std::mt19937_64 g(seed);
    std::uniform_int_distribution<std::int64_t> len(8, 64);
    std::int64_t pos = 1;
    for (std::int64_t s = 0; cells < target_cells; ++s) {
      std::int64_t l = len(g);
      std::int64_t tag = boundary_every > 0 && s % boundary_every == 0 ? 1 : 0;
      if (only_tag < 0 || only_tag == tag) {
        rows.insert(rows.end(), {l, only_tag < 0 ? tag : 0, pos, pos});
        ++count;
      }
      cells += l;
      pos += l + 2;
    }
    extent = pos + 1;
  }
};

} // namespace proto
