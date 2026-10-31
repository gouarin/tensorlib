// PROTOTYPE - correctness of the generated kernels against the hand-written ones, and
// bit-reproducibility of canonical reductions (run it with several OMP_NUM_THREADS).
#include "kernels.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

using namespace proto;

static std::uint64_t bits(double x) { std::uint64_t u; std::memcpy(&u, &x, 8); return u; }

static void compare(const char* what, const double* a, const double* b, std::int64_t n) {
  double worst = 0;
  std::int64_t equal = 0;
  for (std::int64_t i = 0; i < n; ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
    equal += bits(a[i]) == bits(b[i]);
  }
  std::printf("%-28s max|diff| = %.3g, bit-equal %lld / %lld\n", what, worst, (long long)equal, (long long)n);
}

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  jit J;
  std::printf("host cpu: %s, threads: %d\n", J.host_cpu().c_str(), Kokkos::DefaultExecutionSpace().concurrency());
  const std::int64_t n = (1 << 20) + 37;

  {
    arrays a(22, n, 1);
    std::vector<double> args(21);
    for (int s = 0; s <= 20; ++s) args[s] = 0.05 * (s + 1);
    std::vector<double> ref(n);
    auto f = reinterpret_cast<flat_fn>(J.compile(ponio(20)));
    run_flat(f, n, a.data(), args.data());
    Kokkos::fence();
    std::copy(a.ptrs[21], a.ptrs[21] + n, ref.begin());
    hand_ponio20(n, a.data(), args.data());
    Kokkos::fence();
    compare("ponio20 jit vs hand", ref.data(), a.ptrs[21], n);
    auto g = reinterpret_cast<flat_fn>(J.compile(ponio(20), {.noalias = true, .contract = false}));
    run_flat(g, n, a.data(), args.data());
    Kokkos::fence();
    compare("ponio20 jit, no FMA vs FMA", ref.data(), a.ptrs[21], n);
  }

  {
    segments seg(n, 5, 2);
    segments interior(n, 5, 2, 0), boundary(n, 5, 2, 1);
    arrays a(2, seg.extent, 3);
    double args[] = {0.25, 2.0, 0.5};
    std::fill(a.ptrs[1], a.ptrs[1] + seg.extent, 0.0); // gap cells stay 0 in every variant
    auto two = reinterpret_cast<seg_fn>(J.compile(two_body()));
    run_seg(two, seg.count, a.data(), args, seg.rows.data());
    Kokkos::fence();
    std::vector<double> ref(a.ptrs[1], a.ptrs[1] + seg.extent);
    std::fill(a.ptrs[1], a.ptrs[1] + seg.extent, 0.0);
    hand_stencil(seg.count, a.data(), args, seg.rows.data());
    Kokkos::fence();
    compare("two-body jit vs hand", ref.data(), a.ptrs[1], seg.extent);
    std::fill(a.ptrs[1], a.ptrs[1] + seg.extent, 0.0);
    run_seg(reinterpret_cast<seg_fn>(J.compile(stencil_segmented())), interior.count, a.data(), args, interior.rows.data());
    run_seg(reinterpret_cast<seg_fn>(J.compile(boundary_only())), boundary.count, a.data(), args, boundary.rows.data());
    Kokkos::fence();
    compare("two launches vs two-body", ref.data(), a.ptrs[1], seg.extent);
  }

  {
    arrays a(3, n + 1, 4);
    auto f = reinterpret_cast<flat_fn>(J.compile(flux_gather()));
    run_flat(f, n, a.data(), nullptr);
    Kokkos::fence();
    std::vector<double> ref(a.ptrs[2], a.ptrs[2] + n);
    std::copy(a.ptrs[0], a.ptrs[0] + n, a.ptrs[2]);
    hand_flux_scatter(n, a.data());
    Kokkos::fence();
    compare("flux gather vs atomic scatter", ref.data(), a.ptrs[2], n);
  }

  {
    arrays a(1, n, 5);
    std::vector<double> partial;
    auto sum = reinterpret_cast<red_fn>(J.compile_reduction(red_kind::sum, "rsum"));
    auto max = reinterpret_cast<red_fn>(J.compile_reduction(red_kind::max, "rmax"));
    double cs = run_reduce(sum, a.ptrs[0], n, partial, true);
    double cm = run_reduce(max, a.ptrs[0], n, partial, false);
    double ks = kokkos_sum(a.ptrs[0], n);
    long double exact = 0;
    for (std::int64_t i = 0; i < n; ++i) exact += a.ptrs[0][i];
    std::printf("canonical sum %.17g (bits %016llx), max %.17g\n", cs, (unsigned long long)bits(cs), cm);
    std::printf("kokkos    sum %.17g (bits %016llx)\n", ks, (unsigned long long)bits(ks));
    std::printf("long double sum %.17Lg: canonical err %.3g, kokkos err %.3g\n", exact, double(cs - exact), double(ks - exact));
    std::printf("REPRO canonical=%016llx kokkos=%016llx\n", (unsigned long long)bits(cs), (unsigned long long)bits(ks));
  }
}
