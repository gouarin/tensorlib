// PROTOTYPE - throwaway. Correctness of the pending window and graph overhead.
// The bandwidth comparison lives in bench.cpp (Google Benchmark).
#include "tl_window.hpp"

#include <chrono>
#include <cstdio>

using clk = std::chrono::steady_clock;

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard guard(argc, argv);
    std::printf("# threads=%d (Kokkos OpenMP)\n", Kokkos::DefaultExecutionSpace().concurrency());

    // ---- correctness and launch count
    {
        const std::size_t n = 1000;
        tl::tensor a(n), b(n);
        auto av = a.raw(), bv = b.raw();
        Kokkos::parallel_for(n, KOKKOS_LAMBDA(std::size_t i) { av(i) = i; bv(i) = 2.0 * i; });

        tl::st = {};
        tl::tensor c = a + b;
        tl::tensor d = a + 2 * b;
        const double c5 = c(5); // trigger
        std::printf("case 1  c=a+b; d=a+2*b             : c(5)=%g d(5)=%g kernels=%ld instrs=%d slots=%d\n", c5, d(5),
                    tl::st.kernels, tl::st.last_program_size, tl::st.last_scratch_slots);

        tl::st = {};
        auto t = a + b; // tl::expr, never stored
        tl::tensor e = t * t;
        tl::tensor f = t + 2 * b;
        std::printf("case 2  auto t=a+b; e=t*t; f=t+2*b : e(5)=%g f(5)=%g kernels=%ld instrs=%d slots=%d\n", e(5), f(5),
                    tl::st.kernels, tl::st.last_program_size, tl::st.last_scratch_slots);

        tl::st = {};
        tl::tensor g = a + b;
        g = g + 1.0;             // first state dies: inlined, never written
        tl::tensor h = 2 * g;
        std::printf("case 3  g=a+b; g=g+1; h=2*g        : g(5)=%g h(5)=%g kernels=%ld instrs=%d slots=%d\n", g(5), h(5),
                    tl::st.kernels, tl::st.last_program_size, tl::st.last_scratch_slots);
    }

    // ---- cost of building and evaluating the graph (n = 1, so the kernel is negligible)
    {
        tl::tensor a(1), b(1);
        Kokkos::deep_copy(a.raw(), 1.0);
        Kokkos::deep_copy(b.raw(), 2.0);
        const int iters = 200000;

        auto t0 = clk::now();
        for (int it = 0; it < iters; ++it)
        {
            tl::expr e = a + b;
            for (int k = 0; k < 9; ++k) e = e + b; // 1 input pair + 10 op nodes
            (void)e;
            tl::win.nodes.clear();
            tl::win.input_ids.clear();
        }
        const double rec = std::chrono::duration<double, std::nano>(clk::now() - t0).count() / (iters * 12.0);

        double sink = 0;
        t0 = clk::now();
        for (int it = 0; it < iters; ++it)
        {
            tl::tensor c = a + b;
            tl::tensor d = a + 2 * b;
            sink += c(0) + d(0);
        }
        const double cycle = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / iters;

        t0 = clk::now();
        tl::view_t av = a.raw(), bv = b.raw(), cv("c", 1), dv("d", 1);
        for (int it = 0; it < iters; ++it)
        {
            Kokkos::parallel_for(1, KOKKOS_LAMBDA(int i) { cv(i) = av(i) + bv(i); dv(i) = av(i) + 2 * bv(i); });
            Kokkos::fence();
        }
        const double bare = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / iters;
        std::printf("\n# overhead: record %.1f ns/node; full cycle c=a+b;d=a+2*b;read %.2f us; bare fused launch+fence "
                    "%.2f us%s\n",
                    rec, cycle, bare, sink == 42 ? " " : "");
    }
    return 0;
}
