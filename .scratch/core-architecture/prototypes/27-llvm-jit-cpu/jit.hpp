// PROTOTYPE - throwaway code for the "Prototype of the LLVM JIT on CPU" ticket.
// A tiny per-element SSA IR (the shape decided in "Kernel IR"), lowered to LLVM IR
// with IRBuilder and compiled in-process with ORC LLJIT.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace proto {

// ---- mini IR ---------------------------------------------------------------

enum class opk { load, arg, add, sub, mul, max, abs };

struct op {
  opk k;
  int a = -1, b = -1;     // SSA operands
  int operand = -1;       // load: which buffer
  std::int64_t shift = 0; // load: stencil shift (structural)
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

// Flat kernel:      void f(i64 begin, i64 end, double** bufs, const double* args)
// Segmented kernel: void f(i64 sbegin, i64 send, double** bufs, const double* args, const i64* seg)
//   seg row per segment: [length, body tag, base offset of operand 0, ..., operand n-1]
struct kernel {
  std::string name;
  int n_operands = 0;
  bool segmented = false;
  std::vector<body> bodies;
};

using flat_fn = void (*)(std::int64_t, std::int64_t, double**, const double*);
using seg_fn = void (*)(std::int64_t, std::int64_t, double**, const double*, const std::int64_t*);
// Canonical-block reduction: void r(i64 bbegin, i64 bend, const double* a, i64 n, double* partial)
using red_fn = void (*)(std::int64_t, std::int64_t, const double*, std::int64_t, double*);

constexpr std::int64_t canonical_block = 4096; // elements per block, independent of threads and SIMD width
constexpr int canonical_lanes = 8;             // accumulator lanes inside a block

enum class red_kind { sum, max };

struct timings {
  double build_ms = 0, opt_ms = 0, codegen_ms = 0;
  double total() const { return build_ms + opt_ms + codegen_ms; }
};

struct options {
  bool noalias = true;
  bool contract = true; // allow FMA contraction
  bool optimize = true;
};

struct object_code {
  std::string name;
  std::vector<char> bytes;
};

class jit {
public:
  jit();
  ~jit();
  // Builds, optimises and links the kernel; returns its entry point.
  void* compile(const kernel& k, const options& o = {}, timings* t = nullptr);
  void* compile_reduction(red_kind kind, const std::string& name, timings* t = nullptr);
  // Ahead-of-time path: emit an object file, then load it back (the cache-hit path).
  object_code emit_object(const kernel& k, const options& o = {});
  void* load_object(const object_code& obj);
  // Optimised assembly of the kernel, to count vector instructions and spills.
  std::string assembly(const kernel& k, const options& o = {});
  std::string host_cpu() const;

private:
  struct impl;
  std::unique_ptr<impl> p_;
};

} // namespace proto
