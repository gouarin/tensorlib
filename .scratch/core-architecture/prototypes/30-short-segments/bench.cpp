// PROTOTYPE - run time of each lowering on real samurai meshes, with Google Benchmark.
// Usage: bench <intervals file>... [benchmark flags]
#include "plan.hpp"

#include <benchmark/benchmark.h>

#include <deque>

using namespace proto;

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  benchmark::Initialize(&argc, argv);
  jit j;
  std::deque<mesh> meshes;
  std::deque<field_pair> fields;
  std::deque<prepared> preps;
  for (int a = 1; a < argc; ++a) {
    std::string file = argv[a];
    std::string label = file.substr(file.find_last_of('/') + 1);
    for (std::int64_t target : {std::int64_t(0), std::int64_t(1) << 23}) {
      const mesh& m = meshes.emplace_back(load_mesh(file, target));
      field_pair& f = fields.emplace_back(m.extent);
      for (const variant& v : variants()) {
        const prepared& p = preps.emplace_back(j, v, m);
        double** bufs = f.ptrs;
        if (v.sp == space::flat) bufs = new double*[2]{f.u.data() + 8, f.out.data() + 8};
        std::string name = label + "/" + std::to_string(m.cells) + "/" + v.name;
        benchmark::RegisterBenchmark(name, [&p, &m, bufs](benchmark::State& st) {
          for (auto _ : st) {
            p.run(bufs, stencil_args);
            Kokkos::fence();
          }
          st.SetItemsProcessed(st.iterations() * m.cells); // cells of the mesh, whatever the kernel computes
        })->UseRealTime();
      }
    }
  }
  benchmark::RunSpecifiedBenchmarks();
}
