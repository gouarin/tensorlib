// PROTOTYPE - lowering of the mini IR to LLVM IR, optimisation and ORC linking.
#include "jit.hpp"

#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace llvm;

namespace proto {

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

template <class T> T check(Expected<T> e) {
  if (!e) {
    errs() << "LLVM error: " << toString(e.takeError()) << "\n";
    std::abort();
  }
  return std::move(*e);
}

void check(Error e) {
  if (e) {
    errs() << "LLVM error: " << toString(std::move(e)) << "\n";
    std::abort();
  }
}

// Emits one counted loop `for (i = lo; i < hi; ++i) body(i)` and leaves the builder after it.
template <class F> void emit_loop(IRBuilder<>& b, Value* lo, Value* hi, const char* name, F&& body) {
  Function* f = b.GetInsertBlock()->getParent();
  LLVMContext& c = b.getContext();
  BasicBlock* pre = b.GetInsertBlock();
  BasicBlock* head = BasicBlock::Create(c, std::string(name) + ".head", f);
  BasicBlock* loop = BasicBlock::Create(c, std::string(name) + ".body", f);
  BasicBlock* exit = BasicBlock::Create(c, std::string(name) + ".exit", f);
  b.CreateBr(head);
  b.SetInsertPoint(head);
  PHINode* i = b.CreatePHI(b.getInt64Ty(), 2, name);
  i->addIncoming(lo, pre);
  b.CreateCondBr(b.CreateICmpSLT(i, hi), loop, exit);
  b.SetInsertPoint(loop);
  body(static_cast<Value*>(i));
  Value* next = b.CreateAdd(i, b.getInt64(1), "", /*NUW*/ true, /*NSW*/ true);
  i->addIncoming(next, b.GetInsertBlock());
  b.CreateBr(head);
  b.SetInsertPoint(exit);
}

struct lowering {
  Module& m;
  IRBuilder<> b;
  const kernel& k;
  const options& o;
  Type* f64;
  Type* i64;
  Type* ptr;

  lowering(Module& m, const kernel& k, const options& o)
      : m(m), b(m.getContext()), k(k), o(o), f64(b.getDoubleTy()), i64(b.getInt64Ty()), ptr(b.getPtrTy()) {
    FastMathFlags fmf;
    fmf.setAllowContract(o.contract); // FMA contraction only: no reassociation, no NaN/Inf assumptions
    b.setFastMathFlags(fmf);
  }

  // One element of one body at position `i` of a segment (bases are 0 for flat kernels).
  void emit_body(const body& bd, Value* i, const std::vector<Value*>& bufs, const std::vector<Value*>& bases, Value* args) {
    std::vector<Value*> v(bd.ops.size());
    auto index = [&](int operand, std::int64_t shift) {
      Value* idx = b.CreateAdd(bases[operand], i, "", false, true);
      return shift ? b.CreateAdd(idx, b.getInt64(shift), "", false, true) : idx;
    };
    for (std::size_t n = 0; n < bd.ops.size(); ++n) {
      const op& x = bd.ops[n];
      switch (x.k) {
      case opk::load:
        v[n] = b.CreateLoad(f64, b.CreateGEP(f64, bufs[x.operand], index(x.operand, x.shift)));
        break;
      case opk::arg:
        v[n] = b.CreateLoad(f64, b.CreateGEP(f64, args, b.getInt64(x.arg)));
        break;
      case opk::add: v[n] = b.CreateFAdd(v[x.a], v[x.b]); break;
      case opk::sub: v[n] = b.CreateFSub(v[x.a], v[x.b]); break;
      case opk::mul: v[n] = b.CreateFMul(v[x.a], v[x.b]); break;
      case opk::max: v[n] = b.CreateBinaryIntrinsic(Intrinsic::maxnum, v[x.a], v[x.b]); break;
      case opk::abs: v[n] = b.CreateUnaryIntrinsic(Intrinsic::fabs, v[x.a]); break;
      }
    }
    for (const store& s : bd.stores)
      b.CreateStore(v[s.value], b.CreateGEP(f64, bufs[s.operand], index(s.operand, 0)));
  }

  // inner(lo, hi, p0..pn-1, args[, seg]) with noalias buffers, then an exported wrapper
  // taking double** that loads the pointers and calls it (inlining keeps noalias as scopes).
  void emit(const std::string& sym) {
    std::vector<Type*> params{i64, i64};
    for (int j = 0; j < k.n_operands; ++j) params.push_back(ptr);
    params.push_back(ptr); // args
    if (k.segmented) params.push_back(ptr);
    Function* inner = Function::Create(FunctionType::get(b.getVoidTy(), params, false), Function::InternalLinkage, sym + ".inner", m);
    inner->addFnAttr(Attribute::AlwaysInline);
    for (int j = 0; j < k.n_operands; ++j)
      if (o.noalias) inner->addParamAttr(2 + j, Attribute::NoAlias);
    inner->addParamAttr(2 + k.n_operands, Attribute::NoAlias);
    inner->addParamAttr(2 + k.n_operands, Attribute::ReadOnly);

    b.SetInsertPoint(BasicBlock::Create(m.getContext(), "entry", inner));
    std::vector<Value*> bufs;
    for (int j = 0; j < k.n_operands; ++j) bufs.push_back(inner->getArg(2 + j));
    Value* args = inner->getArg(2 + k.n_operands);
    Value* lo = inner->getArg(0);
    Value* hi = inner->getArg(1);
    if (!k.segmented) {
      std::vector<Value*> zero(k.n_operands, b.getInt64(0));
      emit_loop(b, lo, hi, "i", [&](Value* i) { emit_body(k.bodies[0], i, bufs, zero, args); });
    } else {
      Value* seg = inner->getArg(3 + k.n_operands);
      const int row = 2 + k.n_operands;
      emit_loop(b, lo, hi, "s", [&](Value* s) {
        Value* r = b.CreateMul(s, b.getInt64(row));
        auto field = [&](int c) { return b.CreateLoad(i64, b.CreateGEP(i64, seg, b.CreateAdd(r, b.getInt64(c)))); };
        Value* len = field(0);
        std::vector<Value*> bases;
        for (int j = 0; j < k.n_operands; ++j) bases.push_back(field(2 + j));
        if (k.bodies.size() == 1) {
          emit_loop(b, b.getInt64(0), len, "i", [&](Value* i) { emit_body(k.bodies[0], i, bufs, bases, args); });
          return;
        }
        // Several bodies: the segment's tag picks one, uniform over the whole segment.
        Value* tag = field(1);
        Function* f = b.GetInsertBlock()->getParent();
        BasicBlock* join = BasicBlock::Create(m.getContext(), "join", f);
        SwitchInst* sw = b.CreateSwitch(tag, join, unsigned(k.bodies.size()));
        for (std::size_t t = 0; t < k.bodies.size(); ++t) {
          BasicBlock* bb = BasicBlock::Create(m.getContext(), "body" + std::to_string(t), f);
          sw->addCase(b.getInt64(t), bb);
          b.SetInsertPoint(bb);
          emit_loop(b, b.getInt64(0), len, "i", [&](Value* i) { emit_body(k.bodies[t], i, bufs, bases, args); });
          b.CreateBr(join);
        }
        b.SetInsertPoint(join);
      });
    }
    b.CreateRetVoid();

    std::vector<Type*> wparams{i64, i64, ptr, ptr};
    if (k.segmented) wparams.push_back(ptr);
    Function* w = Function::Create(FunctionType::get(b.getVoidTy(), wparams, false), Function::ExternalLinkage, sym, m);
    b.SetInsertPoint(BasicBlock::Create(m.getContext(), "entry", w));
    std::vector<Value*> cargs{w->getArg(0), w->getArg(1)};
    for (int j = 0; j < k.n_operands; ++j)
      cargs.push_back(b.CreateLoad(ptr, b.CreateGEP(ptr, w->getArg(2), b.getInt64(j))));
    cargs.push_back(w->getArg(3));
    if (k.segmented) cargs.push_back(w->getArg(4));
    b.CreateCall(inner, cargs);
    b.CreateRetVoid();
  }
};

// r(bb, be, a, n, partial): for each block, 8 canonical lanes accumulated in element order,
// a fixed tree over the lanes, one partial per block. No reassociation flag anywhere.
void emit_reduction(Module& m, red_kind kind, const std::string& sym) {
  IRBuilder<> b(m.getContext());
  Type* f64 = b.getDoubleTy();
  Type* i64 = b.getInt64Ty();
  Type* ptr = b.getPtrTy();
  auto* vt = FixedVectorType::get(f64, canonical_lanes);
  Function* f = Function::Create(FunctionType::get(b.getVoidTy(), {i64, i64, ptr, i64, ptr}, false), Function::ExternalLinkage, sym, m);
  f->addParamAttr(2, Attribute::NoAlias);
  f->addParamAttr(4, Attribute::NoAlias);
  b.SetInsertPoint(BasicBlock::Create(m.getContext(), "entry", f));
  Value* a = f->getArg(2);
  Value* n = f->getArg(3);
  Value* partial = f->getArg(4);
  double id = kind == red_kind::sum ? 0.0 : -__builtin_inf();
  auto combine = [&](Value* x, Value* y) -> Value* {
    return kind == red_kind::sum ? b.CreateFAdd(x, y) : b.CreateBinaryIntrinsic(Intrinsic::maxnum, x, y);
  };
  emit_loop(b, f->getArg(0), f->getArg(1), "blk", [&](Value* blk) {
    Value* start = b.CreateMul(blk, b.getInt64(canonical_block));
    Value* end = b.CreateBinaryIntrinsic(Intrinsic::smin, b.CreateAdd(start, b.getInt64(canonical_block)), n);
    Value* nvec = b.CreateSDiv(b.CreateSub(end, start), b.getInt64(canonical_lanes));
    AllocaInst* acc = nullptr;
    {
      IRBuilder<> eb(&f->getEntryBlock(), f->getEntryBlock().begin());
      acc = eb.CreateAlloca(vt);
    }
    b.CreateStore(ConstantVector::getSplat(ElementCount::getFixed(canonical_lanes), ConstantFP::get(f64, id)), acc);
    emit_loop(b, b.getInt64(0), nvec, "v", [&](Value* v) {
      Value* j = b.CreateAdd(start, b.CreateMul(v, b.getInt64(canonical_lanes)));
      Value* x = b.CreateAlignedLoad(vt, b.CreateGEP(f64, a, j), Align(8));
      b.CreateStore(combine(b.CreateLoad(vt, acc), x), acc);
    });
    Value* tail = b.CreateAdd(start, b.CreateMul(nvec, b.getInt64(canonical_lanes)));
    emit_loop(b, tail, end, "t", [&](Value* j) {
      Value* lane = b.CreateSub(j, tail);
      Value* cur = b.CreateLoad(vt, acc);
      Value* e = combine(b.CreateExtractElement(cur, lane), b.CreateLoad(f64, b.CreateGEP(f64, a, j)));
      b.CreateStore(b.CreateInsertElement(cur, e, lane), acc);
    });
    Value* v8 = b.CreateLoad(vt, acc);
    Value* v4 = combine(b.CreateShuffleVector(v8, {0, 1, 2, 3}), b.CreateShuffleVector(v8, {4, 5, 6, 7}));
    Value* v2 = combine(b.CreateShuffleVector(v4, {0, 1}), b.CreateShuffleVector(v4, {2, 3}));
    Value* v1 = combine(b.CreateExtractElement(v2, uint64_t(0)), b.CreateExtractElement(v2, uint64_t(1)));
    b.CreateStore(v1, b.CreateGEP(f64, partial, blk));
  });
  b.CreateRetVoid();
}

} // namespace

struct jit::impl {
  orc::JITTargetMachineBuilder jtmb;
  std::unique_ptr<TargetMachine> tm;
  std::unique_ptr<orc::LLJIT> j;
  int counter = 0;

  impl() : jtmb(check(orc::JITTargetMachineBuilder::detectHost())) {
    jtmb.setCodeGenOptLevel(CodeGenOptLevel::Aggressive);
    tm = check(jtmb.createTargetMachine());
    j = check(orc::LLJITBuilder().setJITTargetMachineBuilder(jtmb).create());
  }

  std::unique_ptr<Module> new_module(LLVMContext& c, const std::string& name) {
    auto m = std::make_unique<Module>(name, c);
    m->setDataLayout(tm->createDataLayout());
    m->setTargetTriple(tm->getTargetTriple());
    return m;
  }

  void tag_functions(Module& m) {
    for (Function& f : m) {
      if (f.isDeclaration()) continue;
      f.addFnAttr("target-cpu", tm->getTargetCPU());
      f.addFnAttr("target-features", tm->getTargetFeatureString());
    }
  }

  void optimise(Module& m) {
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;
    PassBuilder pb(tm.get());
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    pb.buildPerModuleDefaultPipeline(OptimizationLevel::O3).run(m, mam);
  }

  std::string emit_file(Module& m, CodeGenFileType type) {
    SmallVector<char, 0> buf;
    raw_svector_ostream os(buf);
    legacy::PassManager pm;
    if (tm->addPassesToEmitFile(pm, os, nullptr, type)) std::abort();
    pm.run(m);
    return std::string(buf.begin(), buf.end());
  }

  // Build and optimise a module for `k` under a fresh symbol; returns it with its context.
  std::pair<orc::ThreadSafeModule, std::string> build(const kernel& k, const options& o, timings* t) {
    auto t0 = std::chrono::steady_clock::now();
    auto ctx = std::make_unique<LLVMContext>();
    std::string sym = k.name + "_" + std::to_string(counter++);
    auto m = new_module(*ctx, sym);
    lowering(*m, k, o).emit(sym);
    tag_functions(*m);
    if (verifyModule(*m, &errs())) std::abort();
    if (t) t->build_ms = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    if (o.optimize) optimise(*m);
    if (t) t->opt_ms = ms_since(t0);
    return {orc::ThreadSafeModule(std::move(m), orc::ThreadSafeContext(std::move(ctx))), sym};
  }

  void* link(orc::ThreadSafeModule tsm, const std::string& sym, timings* t) {
    auto t0 = std::chrono::steady_clock::now();
    check(j->addIRModule(std::move(tsm)));
    void* f = check(j->lookup(sym)).toPtr<void*>();
    if (t) t->codegen_ms = ms_since(t0);
    return f;
  }
};

jit::jit() {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  p_ = std::make_unique<impl>();
}

jit::~jit() = default;

void* jit::compile(const kernel& k, const options& o, timings* t) {
  auto [tsm, sym] = p_->build(k, o, t);
  return p_->link(std::move(tsm), sym, t);
}

void* jit::compile_reduction(red_kind kind, const std::string& name, timings* t) {
  auto t0 = std::chrono::steady_clock::now();
  auto ctx = std::make_unique<LLVMContext>();
  std::string sym = name + "_" + std::to_string(p_->counter++);
  auto m = p_->new_module(*ctx, sym);
  emit_reduction(*m, kind, sym);
  p_->tag_functions(*m);
  if (verifyModule(*m, &errs())) std::abort();
  if (t) t->build_ms = ms_since(t0);
  t0 = std::chrono::steady_clock::now();
  p_->optimise(*m);
  if (t) t->opt_ms = ms_since(t0);
  return p_->link(orc::ThreadSafeModule(std::move(m), orc::ThreadSafeContext(std::move(ctx))), sym, t);
}

object_code jit::emit_object(const kernel& k, const options& o) {
  auto [tsm, sym] = p_->build(k, o, nullptr);
  std::string bytes = tsm.withModuleDo([&](Module& m) { return p_->emit_file(m, CodeGenFileType::ObjectFile); });
  return {sym, std::vector<char>(bytes.begin(), bytes.end())};
}

void* jit::load_object(const object_code& obj) {
  check(p_->j->addObjectFile(MemoryBuffer::getMemBufferCopy(StringRef(obj.bytes.data(), obj.bytes.size()), obj.name)));
  return check(p_->j->lookup(obj.name)).toPtr<void*>();
}

std::string jit::assembly(const kernel& k, const options& o) {
  auto [tsm, sym] = p_->build(k, o, nullptr);
  return tsm.withModuleDo([&](Module& m) { return p_->emit_file(m, CodeGenFileType::AssemblyFile); });
}

std::string jit::host_cpu() const { return p_->tm->getTargetCPU().str(); }

} // namespace proto
