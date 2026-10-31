// PROTOTYPE - cold compile cost per kernel (IR build, O3, codegen + link), object-cache
// reload cost, and what the optimised assembly looks like (vector code, stack spills).
#include "kernels.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

using namespace proto;

constexpr int reps = 9;

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static void cold(jit& J, const char* label, const kernel& k, int ops) {
  std::vector<double> b, o, c, t;
  for (int r = 0; r < reps; ++r) {
    timings tm;
    J.compile(k, {}, &tm);
    b.push_back(tm.build_ms); o.push_back(tm.opt_ms); c.push_back(tm.codegen_ms); t.push_back(tm.total());
  }
  std::printf("%-14s %6d ops   build %7.3f   O3 %8.3f   codegen+link %8.3f   total %8.3f ms\n", label, ops,
              median(b), median(o), median(c), median(t));
}

static int count(const std::string& s, const std::string& what) {
  int n = 0;
  for (std::size_t p = s.find(what); p != std::string::npos; p = s.find(what, p + 1)) ++n;
  return n;
}

// Stack traffic in the kernel: loads and stores addressed off sp (spills and reloads).
static int spills(const std::string& s) {
  int n = 0;
  std::istringstream in(s);
  for (std::string line; std::getline(in, line);)
    if ((line.find("\tstr") != std::string::npos || line.find("\tldr") != std::string::npos ||
         line.find("\tstp") != std::string::npos || line.find("\tldp") != std::string::npos) &&
        line.find("[sp") != std::string::npos)
      ++n;
  return n;
}

static void asm_stats(jit& J, const char* label, const kernel& k, options o = {}) {
  std::string s = J.assembly(k, o);
  std::printf("%-26s vector fmla %3d  vector fmul %3d  vector fadd/fsub %3d  q-reg loads %3d  sp loads/stores %3d\n", label,
              count(s, "fmla\tv") + count(s, "fmls\tv"), count(s, "fmul\tv"), count(s, "fadd\tv") + count(s, "fsub\tv"),
              count(s, "ldr\tq") + count(s, "ldp\tq"), spills(s));
}

static int ops_of(const kernel& k) {
  int n = 0;
  for (const body& b : k.bodies) n += int(b.ops.size());
  return n;
}

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  jit J;
  std::printf("host cpu: %s, median of %d cold compiles, each in a fresh LLVMContext\n\n", J.host_cpu().c_str(), reps);

  std::printf("== reference kernels\n");
  cold(J, "ponio20", ponio(20), ops_of(ponio(20)));
  cold(J, "stencil-seg", stencil_segmented(), ops_of(stencil_segmented()));
  cold(J, "two-body", two_body(), ops_of(two_body()));
  cold(J, "flux-gather", flux_gather(), ops_of(flux_gather()));
  {
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
      timings tm;
      J.compile_reduction(red_kind::sum, "rsum", &tm);
      t.push_back(tm.total());
    }
    std::printf("%-14s %6s        total %8.3f ms\n", "canon-sum", "-", median(t));
  }

  std::printf("\n== growth with kernel size (chain of x = x*s + b)\n");
  for (int steps : {4, 16, 64, 256, 1024, 4096}) cold(J, ("chain" + std::to_string(steps)).c_str(), chain(steps), ops_of(chain(steps)));

  std::printf("\n== growth with live values (L inputs live across 4 outputs)\n");
  for (int L : {4, 8, 16, 24, 32, 48, 64}) cold(J, ("live" + std::to_string(L)).c_str(), live(L), ops_of(live(L)));

  std::printf("\n== object cache: emit once, then reload (file read + ORC link + lookup)\n");
  {
    std::vector<double> emit, reload;
    std::size_t size = 0;
    for (int r = 0; r < reps; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      object_code obj = J.emit_object(ponio(20));
      emit.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/proto27_" + obj.name + ".o";
      std::ofstream(path, std::ios::binary).write(obj.bytes.data(), std::streamsize(obj.bytes.size()));
      size = obj.bytes.size();
      t0 = std::chrono::steady_clock::now();
      std::ifstream in(path, std::ios::binary);
      object_code back{obj.name, std::vector<char>(std::istreambuf_iterator<char>(in), {})};
      J.load_object(back);
      reload.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      std::remove(path.c_str());
    }
    std::printf("ponio20: build+O3+emit object %.3f ms, reload from disk %.3f ms, object size %zu bytes\n", median(emit),
                median(reload), size);
  }

  std::printf("\n== optimised assembly\n");
  asm_stats(J, "ponio20 noalias", ponio(20));
  asm_stats(J, "ponio20 no noalias", ponio(20), {.noalias = false});
  asm_stats(J, "ponio20 no FMA", ponio(20), {.noalias = true, .contract = false});
  asm_stats(J, "stencil-seg", stencil_segmented());
  asm_stats(J, "flux-gather", flux_gather());
  for (int L : {4, 8, 16, 24, 32, 48, 64}) asm_stats(J, ("live" + std::to_string(L)).c_str(), live(L));
  if (argc > 1 && std::string(argv[argc - 1]) == "--dump") std::printf("%s", J.assembly(ponio(20)).c_str());
}
