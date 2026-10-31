// PROTOTYPE - throwaway code for the "Lowering of short segments" ticket.
// The mini IR of prototype 27, cut down to one body, with the lowerings of a segment
// space under comparison: table layout, tail handling and a per-cell index list.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace proto {

enum class opk { load, arg, add, sub, mul };

struct op {
  opk k;
  int a = -1, b = -1;     // SSA operands
  int operand = -1;       // load: which buffer
  std::int64_t shift = 0; // load: stencil shift
  int arg = -1;           // arg: which scalar argument
};

struct store {
  int operand;
  int value;
};

struct body {
  std::vector<op> ops;
  std::vector<store> stores;
  int add(op o) { ops.push_back(o); return int(ops.size()) - 1; }
  int load(int operand, std::int64_t shift = 0) { return add({opk::load, -1, -1, operand, shift}); }
  int arg(int i) { return add({opk::arg, -1, -1, -1, 0, i}); }
  int bin(opk k, int a, int b) { return add({k, a, b}); }
};

// How the iteration space reaches the kernel.
//   flat:  i in [lo, hi), no table (the ceiling).
//   cells: i in [lo, hi), tab[0][i] is the storage offset of cell i.
//   aos:   s in [lo, hi), tab[0] rows of [length, base of operand 0, ..., operand n-1].
//   soa:   s in [lo, hi), tab[0][s] length, tab[1][s] base shared by every operand.
enum class space { flat, cells, aos, soa };

// How a segment is cut into vectors.
//   scalar:  a plain loop per segment, left to LLVM's vectoriser (prototype 27).
//   overlap: whole vectors, the last one shifted back to end at the segment's end;
//            segments shorter than a vector run a scalar loop.
//   masked:  whole vectors, the last one with llvm.masked loads and stores.
//   pad:     whole vectors, the last one runs past the end into the gap after the segment.
enum class tail { scalar, overlap, masked, pad };

struct kernel {
  std::string name;
  int n_operands = 0;
  body bd;
  space sp = space::flat;
  tail tl = tail::scalar;
  int width = 4; // doubles per vector for overlap, masked and pad
};

using fn = void (*)(std::int64_t lo, std::int64_t hi, double** bufs, const double* args, const std::int64_t* const* tab);

class jit {
public:
  jit();
  ~jit();
  fn compile(const kernel& k);
  std::string assembly(const kernel& k);

private:
  struct impl;
  std::unique_ptr<impl> p_;
};

} // namespace proto
