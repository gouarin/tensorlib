// PROTOTYPE - segment tables built at plan time from real samurai meshes, the variants
// under comparison, and their dispatch through Kokkos.
#pragma once
#include "jit.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace proto {

struct seg {
  std::int64_t off, len; // storage offset of the first cell, number of cells
};

// Intervals dumped from samurai (`level y start end index`, offset = index + start),
// in samurai's iteration order, tiled `copies` times to reach a size.
struct mesh {
  std::vector<seg> segs;
  std::int64_t cells = 0, extent = 0; // extent: buffer length including ghosts and padding
};

inline mesh load_mesh(const std::string& file, std::int64_t target_cells) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("cannot open " + file);
  std::vector<seg> one;
  std::int64_t level, y, start, end, index, lo = INT64_MAX, hi = 0, cells = 0;
  while (in >> level >> y >> start >> end >> index) {
    one.push_back({index + start, end - start});
    lo = std::min(lo, index + start);
    hi = std::max(hi, index + end);
    cells += end - start;
  }
  const std::int64_t ghost = 8; // room for the stencil and for whole vectors past the last cell
  const std::int64_t period = hi - lo + 2 * ghost;
  mesh m;
  for (std::int64_t c = 0; m.cells == 0 || m.cells + cells <= std::max(target_cells, cells); ++c) {
    for (seg s : one) m.segs.push_back({s.off - lo + ghost + c * period, s.len});
    m.cells += cells;
    m.extent += period;
  }
  return m;
}

// Merges segments whose storage gap is at most `max_gap` cells: the kernel then also
// computes the gap (ghost) cells, which are dead in the output.
inline std::vector<seg> bridge(std::vector<seg> s, std::int64_t max_gap) {
  std::sort(s.begin(), s.end(), [](seg a, seg b) { return a.off < b.off; });
  std::vector<seg> out;
  for (seg x : s) {
    if (!out.empty() && x.off - (out.back().off + out.back().len) <= max_gap) out.back().len = x.off + x.len - out.back().off;
    else out.push_back(x);
  }
  return out;
}

inline std::vector<seg> sort_by_length(std::vector<seg> s) {
  std::stable_sort(s.begin(), s.end(), [](seg a, seg b) { return a.len < b.len; });
  return s;
}

// Chunk bounds over segments: a fixed count per chunk, or about `cells` cells per chunk.
inline std::vector<std::int64_t> chunks_by_count(std::int64_t n, std::int64_t per) {
  std::vector<std::int64_t> b;
  for (std::int64_t i = 0; i < n; i += per) b.push_back(i);
  b.push_back(n);
  return b;
}

inline std::vector<std::int64_t> chunks_by_cells(const std::vector<seg>& s, std::int64_t cells) {
  std::vector<std::int64_t> b{0};
  std::int64_t acc = 0;
  for (std::int64_t i = 0; i < std::int64_t(s.size()); ++i) {
    acc += s[std::size_t(i)].len;
    if (acc >= cells) {
      b.push_back(i + 1);
      acc = 0;
    }
  }
  if (b.back() != std::int64_t(s.size())) b.push_back(std::int64_t(s.size()));
  return b;
}

// out = u - c * ((u(+1) - two*u) + u(-1)), the stencil of prototype 27. Operands: u, out.
inline body stencil() {
  body b;
  int u = b.load(0);
  int lap = b.bin(opk::add, b.bin(opk::sub, b.load(0, 1), b.bin(opk::mul, b.arg(1), u)), b.load(0, -1));
  b.stores.push_back({1, b.bin(opk::sub, u, b.bin(opk::mul, b.arg(0), lap))});
  return b;
}

enum class transform { none, bridge, sort };
enum class chunking { segs256, one_seg, cells };

struct variant {
  std::string name;
  space sp;
  tail tl = tail::scalar;
  int width = 4;
  transform tr = transform::none;
  chunking ch = chunking::segs256;
};

inline const std::vector<variant>& variants() {
  static const std::vector<variant> v{
      {"flat_ceiling", space::flat},           // a flat loop over as many cells
      {"flat_span", space::flat, tail::scalar, 4, transform::bridge}, // a flat loop over the whole storage span, ghosts included
      {"aos_scalar", space::aos},              // prototype 27
      {"soa_scalar", space::soa},
      {"soa_overlap4", space::soa, tail::overlap, 4},
      {"soa_overlap8", space::soa, tail::overlap, 8},
      {"soa_masked4", space::soa, tail::masked, 4},
      {"soa_pad4", space::soa, tail::pad, 4},
      {"soa_pad2", space::soa, tail::pad, 2},
      {"sorted_overlap4", space::soa, tail::overlap, 4, transform::sort},
      {"bridged_scalar", space::soa, tail::scalar, 4, transform::bridge},
      {"bridged_overlap4", space::soa, tail::overlap, 4, transform::bridge},
      {"bridged_pad4", space::soa, tail::pad, 4, transform::bridge},
      {"overlap4_oneseg", space::soa, tail::overlap, 4, transform::none, chunking::one_seg},
      {"overlap4_cells", space::soa, tail::overlap, 4, transform::none, chunking::cells},
      {"bridged_overlap4_cells", space::soa, tail::overlap, 4, transform::bridge, chunking::cells},
      {"cell_list", space::cells},
  };
  return v;
}

// Cells per parallel_for iteration: 8192 as in prototype 27, smaller on small meshes so
// that every thread gets about four chunks.
inline std::int64_t cell_chunk(std::int64_t cells) {
  return std::clamp<std::int64_t>(cells / (4 * Kokkos::DefaultExecutionSpace().concurrency()), 512, 8192);
}
constexpr std::int64_t max_bridge_gap = 4;  // two ghost cells on each side

// A variant made ready to run on one mesh: compiled kernel, tables and chunk bounds.
struct prepared {
  fn f = nullptr;
  std::vector<std::int64_t> t0, t1, bounds;
  const std::int64_t* tab[2] = {nullptr, nullptr};
  std::int64_t computed = 0; // cells the kernel computes, gap cells included

  prepared(jit& j, const variant& v, const mesh& m) {
    f = j.compile({v.name, 2, stencil(), v.sp, v.tl, v.width});
    std::vector<seg> s = v.tr == transform::bridge ? bridge(m.segs, max_bridge_gap)
                       : v.tr == transform::sort   ? sort_by_length(m.segs)
                                                   : m.segs;
    for (seg x : s) computed += x.len;
    switch (v.sp) {
    case space::flat:
      // transform::bridge on a flat kernel: walk the span from the first to the last cell.
      computed = v.tr == transform::bridge ? m.segs.back().off + m.segs.back().len - m.segs.front().off : m.cells;
      bounds = chunks_by_count(computed, cell_chunk(m.cells));
      break;
    case space::cells:
      for (seg x : s)
        for (std::int64_t i = 0; i < x.len; ++i) t0.push_back(x.off + i);
      bounds = chunks_by_count(std::int64_t(t0.size()), cell_chunk(m.cells));
      break;
    case space::aos:
      for (seg x : s) t0.insert(t0.end(), {x.len, x.off, x.off});
      bounds = chunks_by_count(std::int64_t(s.size()), 256);
      break;
    case space::soa:
      for (seg x : s) {
        t0.push_back(x.len);
        t1.push_back(x.off);
      }
      bounds = v.ch == chunking::segs256 ? chunks_by_count(std::int64_t(s.size()), 256)
             : v.ch == chunking::one_seg ? chunks_by_count(std::int64_t(s.size()), 1)
                                         : chunks_by_cells(s, cell_chunk(m.cells));
      break;
    }
    tab[0] = t0.data();
    tab[1] = t1.data();
  }

  void run(double** bufs, const double* args) const {
    const std::int64_t* b = bounds.data();
    fn g = f;
    const std::int64_t* t0p = tab[0];
    const std::int64_t* t1p = tab[1];
    Kokkos::parallel_for("seg", Kokkos::RangePolicy<>(0, std::int64_t(bounds.size()) - 1), [=](std::int64_t c) {
      const std::int64_t* t[2] = {t0p, t1p};
      g(b[c], b[c + 1], bufs, args, t);
    });
  }
};

struct field_pair {
  std::vector<double> u, out;
  double* ptrs[2];
  explicit field_pair(std::int64_t extent, std::uint64_t seed = 1) : u(std::size_t(extent)), out(std::size_t(extent), 0.0) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (double& x : u) x = d(g);
    ptrs[0] = u.data();
    ptrs[1] = out.data();
  }
};

inline const double stencil_args[2] = {0.3, 2.0};

} // namespace proto
