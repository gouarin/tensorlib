// PROTOTYPE - run time of JIT kernels called from Kokkos against hand-written loops.
#include "kernels.hpp"

#include <benchmark/benchmark.h>

#include <map>

using namespace proto;

namespace {

jit* J = nullptr;
std::map<std::string, void*> fns;

void* fn(const std::string& key, auto make) {
  auto it = fns.find(key);
  if (it == fns.end()) it = fns.emplace(key, make()).first;
  return it->second;
}

void bytes(benchmark::State& st, std::int64_t n, int streams) {
  st.SetBytesProcessed(std::int64_t(st.iterations()) * n * streams * 8);
}

// ---- ponio 20-stage combination -----------------------------------------------------------

void ponio_case(benchmark::State& st, int variant) {
  std::int64_t n = st.range(0);
  static std::map<std::int64_t, arrays*> data;
  if (!data.count(n)) data[n] = new arrays(22, n, 1);
  arrays& a = *data[n];
  std::vector<double> args(21, 0.05);
  flat_fn f = nullptr;
  if (variant == 0) f = reinterpret_cast<flat_fn>(fn("ponio", [] { return J->compile(ponio(20)); }));
  if (variant == 1) f = reinterpret_cast<flat_fn>(fn("ponio-alias", [] { return J->compile(ponio(20), {.noalias = false}); }));
  for (auto _ : st) {
    if (f) run_flat(f, n, a.data(), args.data());
    else hand_ponio20(n, a.data(), args.data());
    Kokkos::fence();
  }
  bytes(st, n, 22);
}

void BM_ponio_jit(benchmark::State& st) { ponio_case(st, 0); }
void BM_ponio_jit_no_noalias(benchmark::State& st) { ponio_case(st, 1); }
void BM_ponio_hand(benchmark::State& st) { ponio_case(st, 2); }

// ---- segmented stencil, one or two bodies --------------------------------------------------

struct seg_data {
  segments all, interior, boundary;
  arrays a;
  explicit seg_data(std::int64_t n) : all(n, 5, 2), interior(n, 5, 2, 0), boundary(n, 5, 2, 1), a(2, all.extent, 3) {}
};

seg_data& segs(std::int64_t n) {
  static std::map<std::int64_t, seg_data*> data;
  if (!data.count(n)) data[n] = new seg_data(n);
  return *data[n];
}

double seg_args[] = {0.25, 2.0, 0.5};

void BM_stencil_jit(benchmark::State& st) {
  seg_data& d = segs(st.range(0));
  auto f = reinterpret_cast<seg_fn>(fn("stencil", [] { return J->compile(stencil_segmented()); }));
  for (auto _ : st) {
    run_seg(f, d.interior.count, d.a.data(), seg_args, d.interior.rows.data());
    Kokkos::fence();
  }
  bytes(st, st.range(0), 2);
}

void BM_two_body_jit_one_launch(benchmark::State& st) {
  seg_data& d = segs(st.range(0));
  auto f = reinterpret_cast<seg_fn>(fn("two", [] { return J->compile(two_body()); }));
  for (auto _ : st) {
    run_seg(f, d.all.count, d.a.data(), seg_args, d.all.rows.data());
    Kokkos::fence();
  }
  bytes(st, st.range(0), 2);
}

void BM_two_body_jit_two_launches(benchmark::State& st) {
  seg_data& d = segs(st.range(0));
  auto f = reinterpret_cast<seg_fn>(fn("stencil", [] { return J->compile(stencil_segmented()); }));
  auto g = reinterpret_cast<seg_fn>(fn("boundary", [] { return J->compile(boundary_only()); }));
  for (auto _ : st) {
    run_seg(f, d.interior.count, d.a.data(), seg_args, d.interior.rows.data());
    run_seg(g, d.boundary.count, d.a.data(), seg_args, d.boundary.rows.data());
    Kokkos::fence();
  }
  bytes(st, st.range(0), 2);
}

void BM_two_body_hand(benchmark::State& st) {
  seg_data& d = segs(st.range(0));
  for (auto _ : st) {
    hand_stencil(d.all.count, d.a.data(), seg_args, d.all.rows.data());
    Kokkos::fence();
  }
  bytes(st, st.range(0), 2);
}

// ---- affine scatter-add: gather form against atomics --------------------------------------

arrays& flux_data(std::int64_t n) {
  static std::map<std::int64_t, arrays*> data;
  if (!data.count(n)) data[n] = new arrays(3, n + 1, 4);
  return *data[n];
}

void BM_flux_jit_gather(benchmark::State& st) {
  std::int64_t n = st.range(0);
  arrays& a = flux_data(n);
  auto f = reinterpret_cast<flat_fn>(fn("flux", [] { return J->compile(flux_gather()); }));
  for (auto _ : st) {
    run_flat(f, n, a.data(), nullptr);
    Kokkos::fence();
  }
  bytes(st, n, 3);
}

void BM_flux_hand_gather(benchmark::State& st) {
  std::int64_t n = st.range(0);
  arrays& a = flux_data(n);
  for (auto _ : st) {
    hand_flux_gather(n, a.data());
    Kokkos::fence();
  }
  bytes(st, n, 3);
}

void BM_flux_hand_atomic_scatter(benchmark::State& st) {
  std::int64_t n = st.range(0);
  arrays& a = flux_data(n);
  for (auto _ : st) {
    hand_flux_scatter(n, a.data()); // accumulates into out; values drift, the cost is what matters
    Kokkos::fence();
  }
  bytes(st, n, 3);
}

// ---- reductions ------------------------------------------------------------------------------

arrays& red_data(std::int64_t n) {
  static std::map<std::int64_t, arrays*> data;
  if (!data.count(n)) data[n] = new arrays(1, n, 5);
  return *data[n];
}

void BM_sum_canonical(benchmark::State& st) {
  std::int64_t n = st.range(0);
  arrays& a = red_data(n);
  std::vector<double> partial;
  auto f = reinterpret_cast<red_fn>(fn("rsum", [] { return J->compile_reduction(red_kind::sum, "rsum"); }));
  for (auto _ : st) benchmark::DoNotOptimize(run_reduce(f, a.ptrs[0], n, partial, true));
  bytes(st, n, 1);
}

void BM_sum_kokkos_free_order(benchmark::State& st) {
  std::int64_t n = st.range(0);
  arrays& a = red_data(n);
  for (auto _ : st) benchmark::DoNotOptimize(kokkos_sum(a.ptrs[0], n));
  bytes(st, n, 1);
}

void BM_max_canonical(benchmark::State& st) {
  std::int64_t n = st.range(0);
  arrays& a = red_data(n);
  std::vector<double> partial;
  auto f = reinterpret_cast<red_fn>(fn("rmax", [] { return J->compile_reduction(red_kind::max, "rmax"); }));
  for (auto _ : st) benchmark::DoNotOptimize(run_reduce(f, a.ptrs[0], n, partial, false));
  bytes(st, n, 1);
}

void BM_max_kokkos_free_order(benchmark::State& st) {
  std::int64_t n = st.range(0);
  arrays& a = red_data(n);
  for (auto _ : st) benchmark::DoNotOptimize(kokkos_max(a.ptrs[0], n));
  bytes(st, n, 1);
}

// ---- live values: time per element as the number of simultaneously live inputs grows ---------

void BM_live(benchmark::State& st) {
  int L = int(st.range(0));
  std::int64_t n = 1 << 18;
  static std::map<int, arrays*> data;
  if (!data.count(L)) data[L] = new arrays(L + 4, n, 6);
  std::vector<double> args(std::size_t(4 * L), 0.01);
  auto f = reinterpret_cast<flat_fn>(fn("live" + std::to_string(L), [&] { return J->compile(live(L)); }));
  for (auto _ : st) {
    run_flat(f, n, data[L]->data(), args.data());
    Kokkos::fence();
  }
  bytes(st, n, L + 4);
  st.counters["ns_per_elem_per_input"] =
      benchmark::Counter(double(n) * (L + 4), benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
}

} // namespace

#define SIZES ->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 23)->UseRealTime()->Unit(benchmark::kMicrosecond)
BENCHMARK(BM_ponio_jit) SIZES;
BENCHMARK(BM_ponio_jit_no_noalias) SIZES;
BENCHMARK(BM_ponio_hand) SIZES;
BENCHMARK(BM_stencil_jit) SIZES;
BENCHMARK(BM_two_body_jit_one_launch) SIZES;
BENCHMARK(BM_two_body_jit_two_launches) SIZES;
BENCHMARK(BM_two_body_hand) SIZES;
BENCHMARK(BM_flux_jit_gather) SIZES;
BENCHMARK(BM_flux_hand_gather) SIZES;
BENCHMARK(BM_flux_hand_atomic_scatter) SIZES;
BENCHMARK(BM_sum_canonical) SIZES;
BENCHMARK(BM_sum_kokkos_free_order) SIZES;
BENCHMARK(BM_max_canonical) SIZES;
BENCHMARK(BM_max_kokkos_free_order) SIZES;
BENCHMARK(BM_live)->Arg(4)->Arg(8)->Arg(16)->Arg(24)->Arg(32)->Arg(48)->Arg(64)->UseRealTime()->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  jit j;
  J = &j;
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}
