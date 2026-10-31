// PROTOTYPE - machine ceilings the kernels are compared with: read bandwidth with K accumulators,
// N read streams + 1 write stream, a flat 3-point stencil, and FMA peak. Plain OpenMP and NEON.
#include <arm_neon.h>
#include <benchmark/benchmark.h>
#include <omp.h>
#include <cstdint>
#include <vector>

using i64 = std::int64_t;
struct bufs : std::vector<double*> { ~bufs() { for (auto q : *this) free(q); } };
static bufs make(int k, i64 n) {
  bufs p;
  for (int j = 0; j < k; ++j) p.push_back(static_cast<double*>(aligned_alloc(64, n * 8 + 64)));
#pragma omp parallel for schedule(static)
  for (i64 i = 0; i < n + 8; ++i) for (int j = 0; j < k; ++j) p[j][i] = 1.0 + 1e-9 * (i % 977) + j;
  return p;
}

template <int K> static void BM_read_acc(benchmark::State& s) {
  i64 n = s.range(0); auto p = make(1, n); const double* a = p[0]; double r = 0;
  for (auto _ : s) {
#pragma omp parallel for schedule(static) reduction(+ : r)
    for (i64 c = 0; c < n; c += 8192) {
      float64x2_t v[K];
      for (int k = 0; k < K; ++k) v[k] = vdupq_n_f64(0);
      for (i64 i = c; i < c + 8192; i += 2 * K)
        for (int k = 0; k < K; ++k) v[k] = vaddq_f64(v[k], vld1q_f64(a + i + 2 * k));
      for (int k = 1; k < K; ++k) v[0] = vaddq_f64(v[0], v[k]);
      r += vaddvq_f64(v[0]);
    }
    benchmark::DoNotOptimize(r);
  }
  s.counters["GB/s"] = benchmark::Counter(8.0 * n, benchmark::Counter::kIsIterationInvariantRate);
}

template <int N> static void BM_streams(benchmark::State& s) {
  i64 n = s.range(0); auto p = make(N + 1, n);
  for (auto _ : s) {
#pragma omp parallel for schedule(static)
    for (i64 c = 0; c < n; c += 8192) {
      const double* q[N];
#pragma GCC unroll 32
      for (int j = 0; j < N; ++j) q[j] = p[j];
      double* o = p[N];
      for (i64 i = c; i < c + 8192; i += 2) {
        float64x2_t a[4] = {vld1q_f64(q[0] + i), vdupq_n_f64(0), vdupq_n_f64(0), vdupq_n_f64(0)};
#pragma GCC unroll 32
        for (int j = 1; j < N; ++j) a[j % 4] = vfmaq_n_f64(a[j % 4], vld1q_f64(q[j] + i), 0.5);
        vst1q_f64(o + i, vaddq_f64(vaddq_f64(a[0], a[1]), vaddq_f64(a[2], a[3])));
      }
    }
    benchmark::ClobberMemory();
  }
  s.counters["GB/s"] = benchmark::Counter(8.0 * (N + 1) * n, benchmark::Counter::kIsIterationInvariantRate);
}

[[gnu::noinline]] static void stencil(i64 lo, i64 hi, double* __restrict o, const double* __restrict u, double c) {
  for (i64 i = lo; i < hi; ++i) o[i] = u[i] - c * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
}
static void BM_flat_stencil(benchmark::State& s) {
  i64 n = s.range(0); auto p = make(2, n);
  for (auto _ : s) {
#pragma omp parallel for schedule(static)
    for (i64 c = 0; c < n; c += 8192) stencil(c == 0 ? 1 : c, c + 8192, p[1], p[0], 0.25);
    benchmark::ClobberMemory();
  }
  s.counters["GB/s"] = benchmark::Counter(16.0 * n, benchmark::Counter::kIsIterationInvariantRate);
}

static void BM_fma_peak16(benchmark::State& s) {
  const i64 iters = 1 << 22; double r = 0;
  for (auto _ : s) {
#pragma omp parallel reduction(+ : r)
    {
      float64x2_t a[16], m = vdupq_n_f64(0.999999), c = vdupq_n_f64(1e-7);
      for (int j = 0; j < 16; ++j) a[j] = vdupq_n_f64(1.0 + j);
      for (i64 i = 0; i < iters; ++i)
        for (int j = 0; j < 16; ++j) a[j] = vfmaq_f64(c, a[j], m);
      for (int j = 0; j < 16; ++j) r += vaddvq_f64(a[j]);
    }
    benchmark::DoNotOptimize(r);
  }
  s.counters["GFlop/s"] = benchmark::Counter(double(iters) * 16 * 4 * omp_get_max_threads(), benchmark::Counter::kIsIterationInvariantRate);
}

#define BIG ->Arg(1 << 23)->UseRealTime()->Unit(benchmark::kMicrosecond)
BENCHMARK(BM_read_acc<4>) BIG; BENCHMARK(BM_read_acc<8>) BIG; BENCHMARK(BM_read_acc<16>) BIG;
BENCHMARK(BM_streams<2>) BIG; BENCHMARK(BM_streams<4>) BIG; BENCHMARK(BM_streams<8>) BIG;
BENCHMARK(BM_streams<12>) BIG; BENCHMARK(BM_streams<16>) BIG; BENCHMARK(BM_streams<21>) BIG;
BENCHMARK(BM_flat_stencil) BIG;
BENCHMARK(BM_fma_peak16)->UseRealTime()->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
