// PROTOTYPE - throwaway. Run time of the generated K=16 expressions, and a check that the
// four variants compute the same values. The tl variant runs the block interpreter of
// tl_lib.cpp (no JIT), so its time is an upper bound for generated code.
#include "et.hpp"
#include "tl_front.hpp"

#include <benchmark/benchmark.h>

#include <xtensor/containers/xtensor.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

using xt2 = xt::xtensor<double, 2>;
constexpr int K = 16, NOUT = 4;

void compute_tl_16(const tl::tensor&, const tl::tensor&, tl::tensor*);
void compute_et_16(const et::tensor&, const et::tensor&, et::tensor*);
void compute_xt_eval_16(const xt2&, const xt2&, xt2*);
void compute_xt_lazy_16(const xt2&, const xt2&, xt2*);

namespace
{
    double xv(std::size_t i, std::size_t n) { return -1.0 + 2.0 * i / n; }
    double yv(std::size_t j, std::size_t m) { return 0.5 - 1.0 * j / m; }

    struct xt_data
    {
        explicit xt_data(std::size_t n) : x(xt2::from_shape({n, 1})), y(xt2::from_shape({1, n}))
        {
            for (std::size_t i = 0; i < n; ++i) { x(i, 0) = xv(i, n); y(0, i) = yv(i, n); }
            for (int o = 0; o < NOUT; ++o) out.push_back(xt2::from_shape({n, n}));
        }
        xt2 x, y;
        std::vector<xt2> out;
    };

    struct et_data
    {
        explicit et_data(std::size_t n) : x(n, 1), y(1, n)
        {
            for (std::size_t i = 0; i < n; ++i) { x.data()[i] = xv(i, n); y.data()[i] = yv(i, n); }
            out.reserve(NOUT);
            for (int o = 0; o < NOUT; ++o) out.emplace_back(n, n);
        }
        et::tensor x, y;
        std::vector<et::tensor> out;
    };

    struct tl_data
    {
        explicit tl_data(std::size_t n) : x(n, 1), y(1, n)
        {
            for (std::size_t i = 0; i < n; ++i) { x.data()[i] = xv(i, n); y.data()[i] = yv(i, n); }
            for (int o = 0; o < NOUT; ++o) out.emplace_back(n, n);
        }
        tl::tensor x, y;
        std::vector<tl::tensor> out;
    };

    void check(std::size_t n)
    {
        xt_data a(n), b(n);
        et_data c(n);
        tl_data d(n);
        compute_xt_lazy_16(a.x, a.y, a.out.data());
        compute_xt_eval_16(b.x, b.y, b.out.data());
        compute_et_16(c.x, c.y, c.out.data());
        compute_tl_16(d.x, d.y, d.out.data());
        double e1 = 0, e2 = 0, e3 = 0, mag = 0;
        for (int o = 0; o < NOUT; ++o)
        {
            const double* pt = d.out[o].data();
            for (std::size_t k = 0; k < n * n; ++k)
            {
                const double ref = a.out[o].data()[k];
                mag = std::max(mag, std::abs(ref));
                e1 = std::max(e1, std::abs(b.out[o].data()[k] - ref));
                e2 = std::max(e2, std::abs(c.out[o].data()[k] - ref));
                e3 = std::max(e3, std::abs(pt[k] - ref));
            }
        }
        std::printf("check n=%zu: max|ref|=%.3g, max diff vs xt_lazy: xt_eval %.2g, et %.2g, tl %.2g\n", n, mag, e1,
                    e2, e3);
    }
} // namespace

static void BM_xt_lazy(benchmark::State& s)
{
    xt_data d(s.range(0));
    for (auto _ : s) { compute_xt_lazy_16(d.x, d.y, d.out.data()); benchmark::ClobberMemory(); }
}
static void BM_xt_eval(benchmark::State& s)
{
    xt_data d(s.range(0));
    for (auto _ : s) { compute_xt_eval_16(d.x, d.y, d.out.data()); benchmark::ClobberMemory(); }
}
static void BM_et(benchmark::State& s)
{
    et_data d(s.range(0));
    for (auto _ : s) { compute_et_16(d.x, d.y, d.out.data()); benchmark::ClobberMemory(); }
}
static void BM_tl(benchmark::State& s)
{
    tl_data d(s.range(0));
    for (auto _ : s) { compute_tl_16(d.x, d.y, d.out.data()); benchmark::ClobberMemory(); }
}
#define SIZES Arg(256)->Arg(2048)->UseRealTime()->Unit(benchmark::kMillisecond)
BENCHMARK(BM_xt_lazy)->SIZES;
BENCHMARK(BM_xt_eval)->SIZES;
BENCHMARK(BM_et)->SIZES;
BENCHMARK(BM_tl)->SIZES;

int main(int argc, char** argv)
{
    check(64);
    check(300);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
