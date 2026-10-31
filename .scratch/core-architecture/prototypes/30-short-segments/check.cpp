// PROTOTYPE - checks every variant against prototype 27's lowering, cell by cell, and
// prints the shape of each mesh's segment table.
#include "plan.hpp"

#include <cstdio>
#include <cstring>

using namespace proto;

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  jit j;
  for (int a = 1; a < argc; ++a) {
    mesh m = load_mesh(argv[a], 0);
    std::vector<seg> br = bridge(m.segs, max_bridge_gap);
    std::int64_t br_cells = 0, short_cells = 0, short_segs = 0, pad_overrun = 0;
    for (seg s : br) br_cells += s.len;
    std::vector<seg> by_off = m.segs;
    std::sort(by_off.begin(), by_off.end(), [](seg x, seg y) { return x.off < y.off; });
    for (std::size_t i = 0; i < by_off.size(); ++i) {
      if (by_off[i].len < 4) { ++short_segs; short_cells += by_off[i].len; }
      std::int64_t padded = (by_off[i].len + 3) & -4;
      if (i + 1 < by_off.size() && by_off[i].off + padded > by_off[i + 1].off) ++pad_overrun;
    }
    std::printf("%s\n  %zu segments, %lld cells, mean length %.1f; %lld segments (%.1f %% of cells) shorter than 4\n",
                argv[a], m.segs.size(), (long long)m.cells, double(m.cells) / double(m.segs.size()), (long long)short_segs,
                100.0 * double(short_cells) / double(m.cells));
    std::printf("  bridged (gap <= %lld): %zu segments, mean length %.1f, %.1f %% extra cells computed\n",
                (long long)max_bridge_gap, br.size(), double(br_cells) / double(br.size()), 100.0 * double(br_cells - m.cells) / double(m.cells));
    std::printf("  pad to 4 runs into the next segment after %lld of %zu segments\n", (long long)pad_overrun, m.segs.size());

    field_pair ref(m.extent);
    const variant& base = *std::find_if(variants().begin(), variants().end(), [](const variant& v) { return v.name == "aos_scalar"; });
    prepared(j, base, m).run(ref.ptrs, stencil_args);
    Kokkos::fence();
    for (const variant& v : variants()) {
      if (v.sp == space::flat) continue;
      field_pair f(m.extent);
      prepared p(j, v, m);
      p.run(f.ptrs, stencil_args);
      Kokkos::fence();
      std::int64_t bad = 0;
      for (seg s : m.segs)
        bad += std::int64_t(std::memcmp(&f.out[std::size_t(s.off)], &ref.out[std::size_t(s.off)], std::size_t(s.len) * sizeof(double)) != 0);
      std::printf("  %-24s %s  computed %lld cells, %zu chunks\n", v.name.c_str(), bad ? "MISMATCH" : "same bits", (long long)p.computed, p.bounds.size() - 1);
    }
  }
}
