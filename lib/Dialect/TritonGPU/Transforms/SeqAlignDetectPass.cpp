//===----------------------------------------------------------------------===//
// SeqAlignDetectPass.cpp
//
// [Context] Shuochen’s hack for sw_kernel v1 — 2025/09/10
//   This pass performs targeted dataflow backtracking to detect the three
//   sequence-alignment ring buffers H/E/F used by `sw_kernel`. It tags matched
//   tt.load/tt.store with an identifying attribute and records lightweight
//   statistics on usage (loads/stores) for each buffer.
//
// [Purpose]
//   - Robustly recover which function arguments (by index) correspond to H/E/F.
//   - Annotate candidate tt.load/tt.store with "seq.buffer" = {"H","E","F"}.
//   - Summarize findings on the `tt.func @sw_kernel` op via attributes.
//
// [Design]
//   - Backtracking starts from each load/store's pointer, follows specific
//     producers (tt.addptr → tt.splat → tt.inttoptr) and falls back to a
//     general operand walk when needed.
//   - Stops at `BlockArgument` that ultimately belongs to a `tt.func`
//     (crossing `scf.while` boundaries by mapping region args back to inits).
//   - H/E/F are identified by fixed positional convention in `sw_kernel`:
//       H_ARG_IDX=5, F_ARG_IDX=6, E_ARG_IDX=7.
//
// [Contract]
//   - Only analyze `tt.func @sw_kernel`.
//   - Treat H/E/F as scalar pointer arguments at the fixed indices above.
//   - Attach `seq.buffer` attributes to matched tt.load/tt.store operations.
//
// [Invariants]
//   - No IR mutation except attaching attributes on matched ops and writing
//     summary attributes on the function.
//   - Safe traversal over `scf.while` by mapping region args to `getInits()`.
//
// [Diagnostics]
//   - Consistent stderr prefix: "[SeqAlignDetect]" with level tags [INFO/WARN/OK/FAIL].
//   - Begin/End banners for easier grep across pipelines.
//
// [Repro]
//   - Enable this pass before lowering; dump IR around tt.load/tt.store sites.
//   - Look for `seq.buffer = "H"|"E"|"F"` and `triton.seqalign.detect` summary.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

/// [Context]
/// Sequence-alignment buffers recognized by convention in `sw_kernel`.
enum class BufKind : uint8_t { Unknown = 0, H, E, F };

// sw_kernel(q, r, m, n, outs, Hbuf, Fbuf, Ebuf, ...)
const int H_ARG_IDX = 5;
const int F_ARG_IDX = 6;
const int E_ARG_IDX = 7;

/// Targeted backtracking to the root function argument.
///
/// [Purpose]
///   Given a pointer-like SSA `Value`, walk backward through well-known Triton
///   producers (tt.addptr / tt.splat / tt.inttoptr) and fall back to a generic
///   operand traversal until we reach a `BlockArgument` owned by `tt.func`.
///
/// [Edge Cases]
///   - If the value flows through `scf.while`, map region args back to `inits`.
///   - Visited-set prevents infinite loops.
///
/// @return
///   The `BlockArgument` that is the ultimate root, or `nullptr` if not found.
static BlockArgument backtrackToFunctionArgument(Value startVal) {
  SmallVector<Value, 16> worklist;
  if (startVal) worklist.push_back(startVal);

  DenseSet<Value> visited;
  if (startVal) visited.insert(startVal);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();

    if (auto barg = llvm::dyn_cast<BlockArgument>(current)) {
      Block *ownerBlock = barg.getOwner();
      if (isa<tt::FuncOp>(ownerBlock->getParentOp())) {
        return barg;
      }

      Operation* parentOp = ownerBlock->getParentOp();
      // NOTE: In dumps, loops may appear as scf.while. Map region-arg -> init.
      if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
        unsigned argIndex = barg.getArgNumber();
        if (argIndex < whileOp.getInits().size()) {
          Value outerOperand = whileOp.getInits()[argIndex];
          if (outerOperand && visited.find(outerOperand) == visited.end()) {
            worklist.push_back(outerOperand);
            visited.insert(outerOperand);
          }
        }
      }
      continue;
    }

    Operation *defOp = current.getDefiningOp();
    if (!defOp) continue;

    // [Design] Prefer targeted producers first; then a generic fallback.
    if (auto addptrOp = llvm::dyn_cast<tt::AddPtrOp>(defOp)) {
      Value basePtr = addptrOp.getPtr();
      if (basePtr && visited.find(basePtr) == visited.end()) {
        worklist.push_back(basePtr);
        visited.insert(basePtr);
      }
    } else if (auto splatOp = llvm::dyn_cast<tt::SplatOp>(defOp)) {
        Value scalarSrc = splatOp.getSrc();
        if (scalarSrc && visited.find(scalarSrc) == visited.end()){
            worklist.push_back(scalarSrc);
            visited.insert(scalarSrc);
        }
    } else if (auto intToPtrOp = llvm::dyn_cast<tt::IntToPtrOp>(defOp)) { 
        Value intSrc = intToPtrOp.getSrc();
        if (intSrc && visited.find(intSrc) == visited.end()){
            worklist.push_back(intSrc);
            visited.insert(intSrc);
        }
    } else { // [Fallback] Explore all operands.
      for (Value operand : defOp->getOperands()) {
        if (operand && visited.find(operand) == visited.end()) {
          worklist.push_back(operand);
          visited.insert(operand);
        }
      }
    }
  }
  return nullptr;
}


/// Pass: SeqAlignDetectPass
///
/// [Pipeline Role]
///   Walk `tt.func @sw_kernel`, detect H/E/F buffer traffic by backtracking,
///   tag matched ops with `seq.buffer`, and attach a compact summary attribute
///   `triton.seqalign.detect` onto the function.
///
/// [Failure Modes]
///   - If `sw_kernel` has insufficient arguments, pass emits a warning and
///     skips analysis.
///   - If no H/E/F traffic is found, emits a FAIL diagnostic.
///
/// [Outputs]
///   - Attributes on matched tt.load/tt.store: `seq.buffer = "H"|"E"|"F"`.
///   - Function-level summary attribute with counts and arg indices.
struct SeqAlignDetectPass
    : public PassWrapper<SeqAlignDetectPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SeqAlignDetectPass)
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tt::TritonDialect, ttg::TritonGPUDialect, scf::SCFDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::errs() << "[SeqAlignDetect][INFO] BEGIN — targeted backtracking\n";

    module.walk([&](tt::FuncOp func) {
      if (func.getName() != "sw_kernel") return;
      llvm::errs() << "[SeqAlignDetect][INFO] Analyzing @" << func.getName() << "\n";

      DenseMap<BlockArgument, BufKind> rootArgMap;
      if (func.getNumArguments() > E_ARG_IDX) {
         rootArgMap[func.getArgument(H_ARG_IDX)] = BufKind::H;
         rootArgMap[func.getArgument(E_ARG_IDX)] = BufKind::E;
         rootArgMap[func.getArgument(F_ARG_IDX)] = BufKind::F;
         llvm::errs() << "[SeqAlignDetect][INFO] Roots — H=" << H_ARG_IDX 
                      << ", E=" << E_ARG_IDX << ", F=" << F_ARG_IDX << "\n";
      } else {
         llvm::errs() << "[SeqAlignDetect][WARN] Function has insufficient arguments to identify H/E/F; skipping.\n";
         return;
      }

      struct BufInfo { int64_t argIndex = -1; int64_t loadCount = 0; int64_t storeCount = 0; };
      BufInfo infoH, infoE, infoF;

      func.walk([&](Operation *op) {
        Value ptr;
        bool isLoad = false;
        if (auto loadOp = llvm::dyn_cast<tt::LoadOp>(op)) {
          ptr = loadOp.getPtr();
          isLoad = true;
        } else if (auto storeOp = llvm::dyn_cast<tt::StoreOp>(op)) {
          ptr = storeOp.getPtr();
        } else {
          return;
        }

        BlockArgument root = backtrackToFunctionArgument(ptr);
        if (!root) return;

        auto it = rootArgMap.find(root);
        if (it == rootArgMap.end()) return;

        BufKind kind = it->second;
        const char* kindStr = (kind == BufKind::H) ? "H" : (kind == BufKind::E) ? "E" : "F";
        op->setAttr("seq.buffer", StringAttr::get(module.getContext(), kindStr));

        if (kind == BufKind::H) {
          if (infoH.argIndex < 0) infoH.argIndex = root.getArgNumber();
          if (isLoad) infoH.loadCount++; else infoH.storeCount++;
        } else if (kind == BufKind::E) {
          if (infoE.argIndex < 0) infoE.argIndex = root.getArgNumber();
          if (isLoad) infoE.loadCount++; else infoE.storeCount++;
        } else if (kind == BufKind::F) {
          if (infoF.argIndex < 0) infoF.argIndex = root.getArgNumber();
          if (isLoad) infoF.loadCount++; else infoF.storeCount++;
        }
      });
      
      auto b = OpBuilder(func);
      auto i64 = [&](int64_t v){ return b.getI64IntegerAttr(v); };
      bool any = (infoH.loadCount + infoH.storeCount + infoE.loadCount + infoE.storeCount + infoF.loadCount + infoF.storeCount) > 0;
      if (any) {
        func->setAttr("triton.seqalign.detect", b.getDictionaryAttr({
            b.getNamedAttr("H", b.getDictionaryAttr({b.getNamedAttr("arg", i64(infoH.argIndex)), b.getNamedAttr("loads", i64(infoH.loadCount)), b.getNamedAttr("stores", i64(infoH.storeCount))})),
            b.getNamedAttr("E", b.getDictionaryAttr({b.getNamedAttr("arg", i64(infoE.argIndex)), b.getNamedAttr("loads", i64(infoE.loadCount)), b.getNamedAttr("stores", i64(infoE.storeCount))})),
            b.getNamedAttr("F", b.getDictionaryAttr({b.getNamedAttr("arg", i64(infoF.argIndex)), b.getNamedAttr("loads", i64(infoF.loadCount)), b.getNamedAttr("stores", i64(infoF.storeCount))}))
        }));
        llvm::errs() << "[SeqAlignDetect][OK] Detected H/E/F buffer traffic in @" << func.getName() << "\n"
                     << "  H: arg(" << infoH.argIndex << "), loads=" << infoH.loadCount << ", stores=" << infoH.storeCount << "\n"
                     << "  E: arg(" << infoE.argIndex << "), loads=" << infoE.loadCount << ", stores=" << infoE.storeCount << "\n"
                     << "  F: arg(" << infoF.argIndex << "), loads=" << infoF.loadCount << ", stores=" << infoF.storeCount << "\n";
      } else {
        llvm::errs() << "[SeqAlignDetect][FAIL] No H/E/F candidates found in @" << func.getName() << "\n";
      }
    });

    llvm::errs() << "[SeqAlignDetect][INFO] END\n";
  }
};
} // namespace

namespace mlir {
std::unique_ptr<mlir::Pass> createSeqAlignDetectPass() {
  return std::make_unique<SeqAlignDetectPass>();
}
} // namespace mlir