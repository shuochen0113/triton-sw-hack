//===----------------------------------------------------------------------===//
// PromoteSeqAlignToShared.cpp
//
// [Context] Shuochen’s hack for sw_kernel v1 — 2025/09/10
//   This pass consumes the H/E/F tags produced by SeqAlignDetect and injects
//   lightweight "anchors" in the IR to mark candidate traffic for shared-memory
//   promotion along the custom SW path.
//
// [Purpose]
//   - Collect tt.load/tt.store carrying `seq.buffer = "H"|"E"|"F"`.
//   - Insert identity `ttg.convert_layout` anchors right at the use sites
//     (for loads) or right before the store, with `smem.anchor = "H"|"E"|"F"`.
//   - Preserve SSA and avoid iterator invalidation while rewriting uses.
//
// [Design]
//   - For a tagged `tt.load` result: wrap each *use* with a ConvertLayoutOp,
//     and replace that single use only (early-inc range to keep iterators safe).
//   - For a tagged `tt.store`: insert a ConvertLayoutOp directly before the
//     store and retarget its value operand to the anchor.
//   - Remove the transient `seq.buffer` tag after anchoring to avoid rework.
//
// [Contract]
//   - Only acts within `tt.func @sw_kernel`.
//   - Requires prior pass to attach `seq.buffer` to tt.load/tt.store.
//   - Anchors are identity converts (type-in = type-out).
//
// [Invariants]
//   - No data layout change intended here; anchors serve as control points for
//     downstream passes to recognize and retarget to SMEM.
//   - The pass does not mutate pointer operands; it only wraps values.
//
// [Diagnostics]
//   - Consistent stderr prefix: `[PromoteSeqAlignToShared][INFO/WARN/OK/FAIL]`.
//   - Begin/End banners ease grepping across pipelines.
//
// [Repro]
//   - Run after SeqAlignDetect; inspect IR for `smem.anchor = "H"|"E"|"F"`
//     on newly inserted `ttg.convert_layout` ops.
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h" // For llvm::make_early_inc_range

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

struct PromoteSeqAlignToShared
    : public PassWrapper<PromoteSeqAlignToShared, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteSeqAlignToShared)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tt::TritonDialect, ttg::TritonGPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::errs() << "[PromoteSeqAlignToShared][INFO] BEGIN\n";

    module.walk([&](tt::FuncOp func) {
      if (func.getName() != "sw_kernel")
        return;

      // [Design] Gather all tt.load/tt.store ops that were pre-tagged.
      SmallVector<Operation*> taggedOps;
      func.walk([&](Operation *op) {
        if (isa<tt::LoadOp, tt::StoreOp>(op) && op->hasAttr("seq.buffer")) {
          taggedOps.push_back(op);
        }
      });
      
      if (taggedOps.empty()) {
        llvm::errs() << "[PromoteSeqAlignToShared][INFO] No tagged ops in @"
                     << func.getName() << " — skipping.\n";
        return;
      }

      llvm::errs() << "[PromoteSeqAlignToShared][INFO] Found "
                   << taggedOps.size() << " tagged ops in @"
                   << func.getName() << " — injecting anchors...\n";

      for (Operation *op : taggedOps) {
        auto bufferKindAttr = op->getAttrOfType<StringAttr>("seq.buffer");
        if (!bufferKindAttr) continue;
        
        // [Contract] Create a stable anchor label ("H" / "E" / "F").
        auto anchorAttr = StringAttr::get(module.getContext(), bufferKindAttr.getValue());

        // ---- Handle tt.load --------------------------------------------------
        if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
          Value oldVal = loadOp.getResult();
          auto oldTy = dyn_cast<RankedTensorType>(oldVal.getType());
          if (!oldTy) continue;

          // [Design] For each single use, insert an identity convert before it
          // and redirect that use only (use early-inc range to avoid invalidation).
          for (OpOperand &use : llvm::make_early_inc_range(oldVal.getUses())) {
            Operation *user = use.getOwner();
            OpBuilder builder(user);
            
            // Identity convert — acts purely as an anchor for later passes.
            auto anchor = builder.create<ttg::ConvertLayoutOp>(
                op->getLoc(), oldTy, oldVal);
            anchor->setAttr("smem.anchor", anchorAttr);

            use.set(anchor.getResult()); // replace this one use
          }
        }
        
        // ---- Handle tt.store -------------------------------------------------
        else if (auto storeOp = dyn_cast<tt::StoreOp>(op)) {
          Value valToStore = storeOp.getValue();
          auto valTy = dyn_cast<RankedTensorType>(valToStore.getType());
          if (!valTy) continue;
          
          OpBuilder builder(storeOp);
          
          // [Design] Insert identity convert right before the store and retarget
          // the store's value operand to the anchor.
          auto anchor = builder.create<ttg::ConvertLayoutOp>(
              op->getLoc(), valTy, valToStore);
          anchor->setAttr("smem.anchor", anchorAttr);
              
          storeOp.getValueMutable().assign(anchor.getResult());
        }

        // [Invariants] Remove transient tag so we don't process it again later.
        op->removeAttr("seq.buffer");
      }
    });

    llvm::errs() << "[PromoteSeqAlignToShared][INFO] END\n";
  }
};

} // namespace

namespace mlir {
std::unique_ptr<mlir::Pass> createPromoteSeqAlignToShared() {
  return std::make_unique<PromoteSeqAlignToShared>();
}
} // namespace mlir