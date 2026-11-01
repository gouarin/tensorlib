// tl::fixed against hand-written loops doing the same arithmetic.
//
// Each kernel runs over an array of `fixed` values, as the body of a loop over a
// tensor<fixed> would, in three versions:
//
// - fixed: the operators of tl::fixed, with their temporaries;
// - loop: the same array of `fixed`, with a hand-written loop over the components of each
//   value. This is the code a user would write without the operators: a gap between
//   `fixed` and `loop` is the cost of the operators;
// - flat: the same scalars as one flat array. It is an upper bound: a gap between `loop`
//   and `flat` comes from the array-of-structures layout, not from tl::fixed.
//
// Every kernel is a noinline function with restrict pointers, so all three give the
// compiler the same aliasing facts and the benchmark loop takes no part in optimising them.
//
// Run with --benchmark_repetitions=10 --benchmark_enable_random_interleaving=true
// --benchmark_report_aggregates_only=true and compare the medians.

#include <tl/tl.hpp>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <vector>

namespace
{
    constexpr std::size_t count = 1024; // values per array: the arrays stay in L2

    template <class F>
    using scalar_of = typename tl::element_traits<F>::scalar;

    template <class F>
    constexpr std::size_t size = tl::detail::product(F::shape());

    namespace kernel
    {
        // out = a + b * c - a / 2: three temporaries per value with the operators.

        template <class F>
        [[gnu::noinline]] void expression_fixed(const F* __restrict a, const F* __restrict b, const F* __restrict c, F* __restrict out)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                out[i] = a[i] + (b[i] * c[i]) - (a[i] / 2);
            }
        }

        template <class F>
        [[gnu::noinline]] void expression_loop(const F* __restrict a, const F* __restrict b, const F* __restrict c, F* __restrict out)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                for (std::size_t k = 0; k < size<F>; ++k)
                {
                    out[i].elements[k] = a[i].elements[k] + (b[i].elements[k] * c[i].elements[k]) - (a[i].elements[k] / 2);
                }
            }
        }

        template <class T, std::size_t S>
        [[gnu::noinline]] void expression_flat(const T* __restrict a, const T* __restrict b, const T* __restrict c, T* __restrict out)
        {
            for (std::size_t i = 0; i < count * S; ++i)
            {
                out[i] = a[i] + (b[i] * c[i]) - (a[i] / 2);
            }
        }

        // out += s * a, in place.

        template <class F>
        [[gnu::noinline]] void compound_fixed(const F* __restrict a, scalar_of<F> s, F* __restrict out)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                out[i] += s * a[i];
            }
        }

        template <class F>
        [[gnu::noinline]] void compound_loop(const F* __restrict a, scalar_of<F> s, F* __restrict out)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                for (std::size_t k = 0; k < size<F>; ++k)
                {
                    out[i].elements[k] += s * a[i].elements[k];
                }
            }
        }

        template <class T, std::size_t S>
        [[gnu::noinline]] void compound_flat(const T* __restrict a, T s, T* __restrict out)
        {
            for (std::size_t i = 0; i < count * S; ++i)
            {
                out[i] += s * a[i];
            }
        }
    }

    template <class F>
    std::vector<F> values(std::size_t seed)
    {
        std::vector<F> result(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            for (std::size_t k = 0; k < size<F>; ++k)
            {
                result[i].elements[k] = static_cast<scalar_of<F>>(1 + ((seed + i + k) % 7));
            }
        }
        return result;
    }

    // The scalars of an array of fixed, as one flat array.
    template <class F>
    std::vector<scalar_of<F>> flat(const std::vector<F>& fixed)
    {
        std::vector<scalar_of<F>> result;
        for (const F& value : fixed)
        {
            result.insert(result.end(), value.elements.begin(), value.elements.end());
        }
        return result;
    }

    template <class F>
    void report(benchmark::State& state)
    {
        state.SetItemsProcessed(state.iterations() * static_cast<benchmark::IterationCount>(count * size<F>));
    }

    // Runs kernel(a, b, c, out) over arrays built by `make` from values<F>.
    template <class F, class Make, class Kernel>
    void run_expression(benchmark::State& state, Make make, Kernel kernel)
    {
        const auto a = make(values<F>(0));
        const auto b = make(values<F>(1));
        const auto c = make(values<F>(2));
        auto out     = make(values<F>(3));
        for (auto _ : state)
        {
            kernel(a.data(), b.data(), c.data(), out.data());
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
        report<F>(state);
    }

    // Runs kernel(a, s, out) over arrays built by `make` from values<F>.
    template <class F, class Make, class Kernel>
    void run_compound(benchmark::State& state, Make make, Kernel kernel)
    {
        const auto a = make(values<F>(0));
        auto out     = make(values<F>(1));
        for (auto _ : state)
        {
            kernel(a.data(), scalar_of<F>{1} / 1024, out.data());
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
        report<F>(state);
    }

    template <class F>
    std::vector<F> same(std::vector<F> values)
    {
        return values;
    }

    template <class F>
    void expression_fixed(benchmark::State& state)
    {
        run_expression<F>(state, same<F>, kernel::expression_fixed<F>);
    }

    template <class F>
    void expression_loop(benchmark::State& state)
    {
        run_expression<F>(state, same<F>, kernel::expression_loop<F>);
    }

    template <class F>
    void expression_flat(benchmark::State& state)
    {
        run_expression<F>(state, flat<F>, kernel::expression_flat<scalar_of<F>, size<F>>);
    }

    template <class F>
    void compound_fixed(benchmark::State& state)
    {
        run_compound<F>(state, same<F>, kernel::compound_fixed<F>);
    }

    template <class F>
    void compound_loop(benchmark::State& state)
    {
        run_compound<F>(state, same<F>, kernel::compound_loop<F>);
    }

    template <class F>
    void compound_flat(benchmark::State& state)
    {
        run_compound<F>(state, flat<F>, kernel::compound_flat<scalar_of<F>, size<F>>);
    }

    using vec3  = tl::fixed<double, 3>;             // 24 bytes
    using mat3  = tl::fixed<double, 3, 3>;          // 72 bytes
    using vec8f = tl::fixed<float, 8>;              // 32 bytes
    using mat4  = tl::fixed<double, 4, 4>;          // 128 bytes, unrolled on every target
    using rank5 = tl::fixed<double, 2, 2, 2, 2, 2>; // 256 bytes, unrolled with AVX or on Arm, a loop on SSE2
    using mat6  = tl::fixed<double, 6, 6>;          // 288 bytes, a loop except with AVX-512
}

BENCHMARK_TEMPLATE(expression_fixed, vec3);
BENCHMARK_TEMPLATE(expression_loop, vec3);
BENCHMARK_TEMPLATE(expression_flat, vec3);
BENCHMARK_TEMPLATE(compound_fixed, vec3);
BENCHMARK_TEMPLATE(compound_loop, vec3);
BENCHMARK_TEMPLATE(compound_flat, vec3);

BENCHMARK_TEMPLATE(expression_fixed, mat3);
BENCHMARK_TEMPLATE(expression_loop, mat3);
BENCHMARK_TEMPLATE(expression_flat, mat3);
BENCHMARK_TEMPLATE(compound_fixed, mat3);
BENCHMARK_TEMPLATE(compound_loop, mat3);
BENCHMARK_TEMPLATE(compound_flat, mat3);

BENCHMARK_TEMPLATE(expression_fixed, vec8f);
BENCHMARK_TEMPLATE(expression_loop, vec8f);
BENCHMARK_TEMPLATE(expression_flat, vec8f);
BENCHMARK_TEMPLATE(compound_fixed, vec8f);
BENCHMARK_TEMPLATE(compound_loop, vec8f);
BENCHMARK_TEMPLATE(compound_flat, vec8f);

BENCHMARK_TEMPLATE(expression_fixed, mat4);
BENCHMARK_TEMPLATE(expression_loop, mat4);
BENCHMARK_TEMPLATE(expression_flat, mat4);
BENCHMARK_TEMPLATE(compound_fixed, mat4);
BENCHMARK_TEMPLATE(compound_loop, mat4);
BENCHMARK_TEMPLATE(compound_flat, mat4);

BENCHMARK_TEMPLATE(expression_fixed, rank5);
BENCHMARK_TEMPLATE(expression_loop, rank5);
BENCHMARK_TEMPLATE(expression_flat, rank5);
BENCHMARK_TEMPLATE(compound_fixed, rank5);
BENCHMARK_TEMPLATE(compound_loop, rank5);
BENCHMARK_TEMPLATE(compound_flat, rank5);

BENCHMARK_TEMPLATE(expression_fixed, mat6);
BENCHMARK_TEMPLATE(expression_loop, mat6);
BENCHMARK_TEMPLATE(expression_flat, mat6);
BENCHMARK_TEMPLATE(compound_fixed, mat6);
BENCHMARK_TEMPLATE(compound_loop, mat6);
BENCHMARK_TEMPLATE(compound_flat, mat6);
