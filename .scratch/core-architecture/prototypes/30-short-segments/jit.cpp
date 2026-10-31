// PROTOTYPE - lowering of the mini IR to LLVM IR for the segment-space variants.
#include "jit.hpp"

#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <cstdlib>

using namespace llvm;

namespace proto {

namespace {

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

// Emits `for (i = lo; i < hi; i += step) body(i)`. `vectorise = false` tags the loop so
// that LLVM leaves it scalar (used for loops known to run fewer times than a vector).
template <class F>
void emit_loop(IRBuilder<>& b, Value* lo, Value* hi, const char* name, F&& body, std::int64_t step = 1, bool vectorise = true) {
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
  Value* next = b.CreateAdd(i, b.getInt64(step), "", true, true);
  i->addIncoming(next, b.GetInsertBlock());
  BranchInst* back = b.CreateBr(head);
  if (!vectorise) {
    MDNode* off = MDNode::get(c, {MDString::get(c, "llvm.loop.vectorize.enable"), ConstantAsMetadata::get(b.getFalse())});
    MDNode* unroll = MDNode::get(c, {MDString::get(c, "llvm.loop.unroll.disable")});
    MDNode* id = MDNode::getDistinct(c, {nullptr, off, unroll});
    id->replaceOperandWith(0, id);
    back->setMetadata(LLVMContext::MD_loop, id);
  }
  b.SetInsertPoint(exit);
}

struct lowering {
  Module& m;
  IRBuilder<> b;
  const kernel& k;
  Type* f64;
  Type* i64;
  Type* ptr;

  lowering(Module& m, const kernel& k) : m(m), b(m.getContext()), k(k), f64(b.getDoubleTy()), i64(b.getInt64Ty()), ptr(b.getPtrTy()) {
    FastMathFlags fmf;
    fmf.setAllowContract(true); // FMA only, as decided in "Code-generation strategy"
    b.setFastMathFlags(fmf);
  }

  // One body at positions bases[j] + i (+ shift), `w` lanes wide (w == 1: scalar).
  // `mask`, when set, guards every load and store of the vector.
  void emit_body(Value* i, const std::vector<Value*>& bufs, const std::vector<Value*>& bases, Value* args, int w, Value* mask = nullptr) {
    Type* vt = w == 1 ? f64 : static_cast<Type*>(FixedVectorType::get(f64, w));
    auto at = [&](int operand, std::int64_t shift) {
      Value* idx = b.CreateAdd(bases[operand], i, "", false, true);
      if (shift) idx = b.CreateAdd(idx, b.getInt64(shift), "", false, true);
      return b.CreateGEP(f64, bufs[operand], idx);
    };
    std::vector<Value*> v(k.bd.ops.size());
    for (std::size_t n = 0; n < k.bd.ops.size(); ++n) {
      const op& x = k.bd.ops[n];
      switch (x.k) {
      case opk::load:
        v[n] = mask ? static_cast<Value*>(b.CreateMaskedLoad(vt, at(x.operand, x.shift), Align(8), mask))
                    : b.CreateAlignedLoad(vt, at(x.operand, x.shift), Align(8));
        break;
      case opk::arg: {
        Value* s = b.CreateLoad(f64, b.CreateGEP(f64, args, b.getInt64(x.arg)));
        v[n] = w == 1 ? s : b.CreateVectorSplat(w, s);
        break;
      }
      case opk::add: v[n] = b.CreateFAdd(v[x.a], v[x.b]); break;
      case opk::sub: v[n] = b.CreateFSub(v[x.a], v[x.b]); break;
      case opk::mul: v[n] = b.CreateFMul(v[x.a], v[x.b]); break;
      }
    }
    for (const store& s : k.bd.stores) {
      if (mask) b.CreateMaskedStore(v[s.value], at(s.operand, 0), Align(8), mask);
      else b.CreateAlignedStore(v[s.value], at(s.operand, 0), Align(8));
    }
  }

  // One segment of length `len` whose operands start at `bases`.
  void emit_segment(Value* len, const std::vector<Value*>& bufs, const std::vector<Value*>& bases, Value* args) {
    const int w = k.width;
    Function* f = b.GetInsertBlock()->getParent();
    LLVMContext& c = m.getContext();
    switch (k.tl) {
    case tail::scalar:
      emit_loop(b, b.getInt64(0), len, "i", [&](Value* i) { emit_body(i, bufs, bases, args, 1); });
      return;
    case tail::pad: {
      // ceil(len / w) whole vectors; the last one reads and writes the gap after the segment.
      Value* end = b.CreateAnd(b.CreateAdd(len, b.getInt64(w - 1)), b.getInt64(-w));
      emit_loop(b, b.getInt64(0), end, "v", [&](Value* i) { emit_body(i, bufs, bases, args, w); }, w, false);
      return;
    }
    case tail::masked: {
      Value* end = b.CreateAnd(b.CreateAdd(len, b.getInt64(w - 1)), b.getInt64(-w));
      std::vector<Constant*> lanes;
      for (int l = 0; l < w; ++l) lanes.push_back(b.getInt64(l));
      Value* iota = ConstantVector::get(lanes);
      Value* lenv = b.CreateVectorSplat(w, len);
      emit_loop(b, b.getInt64(0), end, "v", [&](Value* i) {
        Value* mask = b.CreateICmpSLT(b.CreateAdd(b.CreateVectorSplat(w, i), iota), lenv);
        emit_body(i, bufs, bases, args, w, mask);
      }, w, false);
      return;
    }
    case tail::overlap: {
      // Whole vectors, then one vector ending exactly at len, or a scalar loop when len < w.
      Value* whole = b.CreateAnd(len, b.getInt64(-w));
      emit_loop(b, b.getInt64(0), whole, "v", [&](Value* i) { emit_body(i, bufs, bases, args, w); }, w, false);
      BasicBlock* has_tail = BasicBlock::Create(c, "has_tail", f);
      BasicBlock* last = BasicBlock::Create(c, "last_vector", f);
      BasicBlock* scalar = BasicBlock::Create(c, "short", f);
      BasicBlock* done = BasicBlock::Create(c, "seg_done", f);
      b.CreateCondBr(b.CreateICmpNE(whole, len), has_tail, done);
      b.SetInsertPoint(has_tail);
      b.CreateCondBr(b.CreateICmpSGE(len, b.getInt64(w)), last, scalar);
      b.SetInsertPoint(last);
      emit_body(b.CreateSub(len, b.getInt64(w)), bufs, bases, args, w);
      b.CreateBr(done);
      b.SetInsertPoint(scalar);
      emit_loop(b, b.getInt64(0), len, "t", [&](Value* i) { emit_body(i, bufs, bases, args, 1); }, 1, false);
      b.CreateBr(done);
      b.SetInsertPoint(done);
      return;
    }
    }
  }

  // inner(lo, hi, p0..pn-1, args, tab0, tab1) with noalias buffers, inlined into an exported
  // wrapper taking double** and i64** (inlining keeps noalias as scopes).
  void emit(const std::string& sym) {
    std::vector<Type*> params{i64, i64};
    for (int j = 0; j < k.n_operands; ++j) params.push_back(ptr);
    params.insert(params.end(), {ptr, ptr, ptr}); // args, tab0, tab1
    Function* inner = Function::Create(FunctionType::get(b.getVoidTy(), params, false), Function::InternalLinkage, sym + ".inner", m);
    inner->addFnAttr(Attribute::AlwaysInline);
    for (int j = 0; j < k.n_operands + 3; ++j) inner->addParamAttr(2 + j, Attribute::NoAlias);
    b.SetInsertPoint(BasicBlock::Create(m.getContext(), "entry", inner));
    std::vector<Value*> bufs;
    for (int j = 0; j < k.n_operands; ++j) bufs.push_back(inner->getArg(2 + j));
    Value* args = inner->getArg(2 + k.n_operands);
    Value* t0 = inner->getArg(3 + k.n_operands);
    Value* t1 = inner->getArg(4 + k.n_operands);
    Value* lo = inner->getArg(0);
    Value* hi = inner->getArg(1);
    auto entry = [&](Value* t, Value* idx) { return b.CreateLoad(i64, b.CreateGEP(i64, t, idx)); };

    switch (k.sp) {
    case space::flat: {
      std::vector<Value*> zero(k.n_operands, b.getInt64(0));
      emit_loop(b, lo, hi, "i", [&](Value* i) { emit_body(i, bufs, zero, args, 1); });
      break;
    }
    case space::cells:
      emit_loop(b, lo, hi, "c", [&](Value* c) {
        std::vector<Value*> base(k.n_operands, entry(t0, c));
        emit_body(b.getInt64(0), bufs, base, args, 1);
      });
      break;
    case space::aos: {
      const int row = 1 + k.n_operands;
      emit_loop(b, lo, hi, "s", [&](Value* s) {
        Value* r = b.CreateMul(s, b.getInt64(row));
        std::vector<Value*> bases;
        for (int j = 0; j < k.n_operands; ++j) bases.push_back(entry(t0, b.CreateAdd(r, b.getInt64(1 + j))));
        emit_segment(entry(t0, r), bufs, bases, args);
      });
      break;
    }
    case space::soa:
      emit_loop(b, lo, hi, "s", [&](Value* s) {
        std::vector<Value*> bases(k.n_operands, entry(t1, s));
        emit_segment(entry(t0, s), bufs, bases, args);
      });
      break;
    }
    b.CreateRetVoid();

    Function* w = Function::Create(FunctionType::get(b.getVoidTy(), {i64, i64, ptr, ptr, ptr}, false), Function::ExternalLinkage, sym, m);
    b.SetInsertPoint(BasicBlock::Create(m.getContext(), "entry", w));
    std::vector<Value*> cargs{w->getArg(0), w->getArg(1)};
    for (int j = 0; j < k.n_operands; ++j) cargs.push_back(b.CreateLoad(ptr, b.CreateGEP(ptr, w->getArg(2), b.getInt64(j))));
    cargs.push_back(w->getArg(3));
    for (int t = 0; t < 2; ++t) cargs.push_back(b.CreateLoad(ptr, b.CreateGEP(ptr, w->getArg(4), b.getInt64(t))));
    b.CreateCall(inner, cargs);
    b.CreateRetVoid();
  }
};

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

  std::pair<orc::ThreadSafeModule, std::string> build(const kernel& k) {
    auto ctx = std::make_unique<LLVMContext>();
    std::string sym = k.name + "_" + std::to_string(counter++);
    auto m = std::make_unique<Module>(sym, *ctx);
    m->setDataLayout(tm->createDataLayout());
    m->setTargetTriple(tm->getTargetTriple());
    lowering(*m, k).emit(sym);
    for (Function& f : *m) {
      if (f.isDeclaration()) continue;
      f.addFnAttr("target-cpu", tm->getTargetCPU());
      f.addFnAttr("target-features", tm->getTargetFeatureString());
    }
    if (verifyModule(*m, &errs())) std::abort();
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
    pb.buildPerModuleDefaultPipeline(OptimizationLevel::O3).run(*m, mam);
    return {orc::ThreadSafeModule(std::move(m), orc::ThreadSafeContext(std::move(ctx))), sym};
  }
};

jit::jit() {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  p_ = std::make_unique<impl>();
}

jit::~jit() = default;

fn jit::compile(const kernel& k) {
  auto [tsm, sym] = p_->build(k);
  check(p_->j->addIRModule(std::move(tsm)));
  return check(p_->j->lookup(sym)).toPtr<fn>();
}

std::string jit::assembly(const kernel& k) {
  auto [tsm, sym] = p_->build(k);
  return tsm.withModuleDo([&](Module& m) {
    SmallVector<char, 0> buf;
    raw_svector_ostream os(buf);
    legacy::PassManager pm;
    if (p_->tm->addPassesToEmitFile(pm, os, nullptr, CodeGenFileType::AssemblyFile)) std::abort();
    pm.run(m);
    return std::string(buf.begin(), buf.end());
  });
}

} // namespace proto
