// PROTOTYPE - throwaway. Bandwidth and overhead of the pending window, with Google Benchmark.
//
// Run with repetitions and random interleaving (see run.sh): on a noisy machine, variants
// measured back to back compare different machine states.

#include "tl_window.hpp"

#include <benchmark/benchmark.h>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/core/xnoalias.hpp>

#include <map>
#include <memory>

namespace
{
    using xt1d = xt::xtensor<double, 1>;

    struct data // allocated once per size, shared by all variants
    {
        explicit data(std::size_t n)
            : a("a", n), b("b", n), c("c", n), d("d", n), ta(n), tb(n), tc(n), td(n),
              xa(xt1d::from_shape({n})), xb(xa), xc(xa), xd(xa)
        {
            auto av = a, bv = b;
            Kokkos::parallel_for(n, KOKKOS_LAMBDA(std::size_t i) { av(i) = 1.0 + i % 7; bv(i) = 2.0 + i % 5; });
            Kokkos::deep_copy(ta.raw(), a);
            Kokkos::deep_copy(tb.raw(), b);
            tc = ta + tb;
            td = ta + 2 * tb;
            tl::evaluate();
            for (std::size_t i = 0; i < n; ++i) { xa(i) = 1.0 + i % 7; xb(i) = 2.0 + i % 5; }
            xc = xa;
            xd = xa;
            Kokkos::fence();
        }

        tl::view_t a, b, c, d;
        tl::tensor ta, tb, tc, td;
        xt1d xa, xb, xc, xd;
    };

    std::map<std::size_t, std::unique_ptr<data>>& cache()
    {
        static std::map<std::size_t, std::unique_ptr<data>> m;
        return m;
    }

    data& get(std::size_t n)
    {
        auto& p = cache()[n];
        if (!p) p = std::make_unique<data>(n);
        return *p;
    }

    // Same logical traffic for every variant (read a, b; write c, d), so the rates compare directly.
    void finish(benchmark::State& s, std::size_t n)
    {
        s.SetBytesProcessed(static_cast<int64_t>(s.iterations()) * 4 * n * sizeof(double));
    }
} // namespace

static void two_loops(benchmark::State& s)
{
    const std::size_t n = s.range(0);
    auto& x = get(n);
    auto a = x.a, b = x.b, c = x.c, d = x.d;
    for (auto _ : s)
    {
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(std::size_t i) { c(i) = a(i) + b(i); });
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(std::size_t i) { d(i) = a(i) + 2.0 * b(i); });
        Kokkos::fence();
    }
    finish(s, n);
}

static void fused_by_hand(benchmark::State& s)
{
    const std::size_t n = s.range(0);
    auto& x = get(n);
    auto a = x.a, b = x.b, c = x.c, d = x.d;
    for (auto _ : s)
    {
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(std::size_t i) {
            c(i) = a(i) + b(i);
            d(i) = a(i) + 2.0 * b(i);
        });
        Kokkos::fence();
    }
    finish(s, n);
}

static void window_fresh_alloc(benchmark::State& s)
{
    const std::size_t n = s.range(0);
    auto& x = get(n);
    tl::reuse_buffers = false;
    for (auto _ : s)
    {
        tl::tensor c = x.ta + x.tb;
        tl::tensor d = x.ta + 2 * x.tb;
        tl::evaluate();
        Kokkos::fence();
        benchmark::DoNotOptimize(c);
        benchmark::DoNotOptimize(d);
    }
    tl::reuse_buffers = true;
    finish(s, n);
}

static void window_reuse(benchmark::State& s) // time loop: the same handles reassigned each step
{
    const std::size_t n = s.range(0);
    auto& x = get(n);
    for (auto _ : s)
    {
        x.tc = x.ta + x.tb;
        x.td = x.ta + 2 * x.tb;
        tl::evaluate();
        Kokkos::fence();
    }
    finish(s, n);
}

static void xtensor_assign(benchmark::State& s)
{
    const std::size_t n = s.range(0);
    auto& x = get(n);
    for (auto _ : s)
    {
        x.xc = x.xa + x.xb;
        x.xd = x.xa + 2.0 * x.xb;
        benchmark::ClobberMemory();
    }
    finish(s, n);
}

static void xtensor_noalias(benchmark::State& s)
{
    const std::size_t n = s.range(0);
    auto& x = get(n);
    for (auto _ : s)
    {
        xt::noalias(x.xc) = x.xa + x.xb;
        xt::noalias(x.xd) = x.xa + 2.0 * x.xb;
        benchmark::ClobberMemory();
    }
    finish(s, n);
}

#define SIZES RangeMultiplier(16)->Range(1 << 12, 1 << 24)->Arg(1 << 26)->UseRealTime()->Unit(benchmark::kMicrosecond)
BENCHMARK(two_loops)->SIZES;
BENCHMARK(fused_by_hand)->SIZES;
BENCHMARK(window_fresh_alloc)->SIZES;
BENCHMARK(window_reuse)->SIZES;
BENCHMARK(xtensor_assign)->SIZES;
BENCHMARK(xtensor_noalias)->SIZES;

// ---- overhead of the window itself

static void record_nodes(benchmark::State& s) // 2 inputs + 10 operations per iteration
{
    tl::tensor a(1), b(1);
    for (auto _ : s)
    {
        tl::expr e = a + b;
        for (int k = 0; k < 9; ++k) e = e + b;
        benchmark::DoNotOptimize(e);
        tl::win.nodes.clear();
        tl::win.input_ids.clear();
    }
    s.SetItemsProcessed(static_cast<int64_t>(s.iterations()) * 12);
}
BENCHMARK(record_nodes);

static void window_cycle_n1(benchmark::State& s) // record 2 statements, evaluate, wait
{
    auto& x = get(1);
    for (auto _ : s)
    {
        x.tc = x.ta + x.tb;
        x.td = x.ta + 2 * x.tb;
        tl::evaluate();
        Kokkos::fence();
    }
}
BENCHMARK(window_cycle_n1)->UseRealTime();

static void bare_launch_n1(benchmark::State& s) // the kernel launch alone
{
    auto& x = get(1);
    auto a = x.a, b = x.b, c = x.c, d = x.d;
    for (auto _ : s)
    {
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(int i) {
            c(i) = a(i) + b(i);
            d(i) = a(i) + 2.0 * b(i);
        });
        Kokkos::fence();
    }
}
BENCHMARK(bare_launch_n1)->UseRealTime();

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard guard(argc, argv);
    benchmark::Initialize(&argc, argv);
    benchmark::AddCustomContext("kokkos_concurrency", std::to_string(Kokkos::DefaultExecutionSpace().concurrency()));
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    cache().clear(); // free Kokkos views before Kokkos::finalize
    return 0;
}
