//===----------------------------------------------------------------------===//
// MaterializeSWSmem.cpp
//
// [Context] Shuochen’s hack for sw_kernel v1 — 2025/09/10
//   This is the *dispatcher* pass in the custom SW path. It materializes H/E/F
//   ring buffers in shared memory (SMEM), routed from previously inserted
//   anchors, and rewrites load/store traffic into local_*_slice ops that carry
//   (MemDesc, OffsetTensor) *losslessly* to lowering.
//
// [Purpose]
//   - Scan for `ttg.convert_layout` anchors labeled with `smem.anchor = "H"|"E"|"F"`.
//   - Allocate SMEM memdescs with linear_shared encoding (2D [SLOTS, STRIDE]).
//   - Initialize SMEM buffers with sentinel values (e.g., -1e7 for i32).
//   - Derive ring slot index from base pointer arithmetic and compute
//     per-thread offset tensors from the anchor’s original pointer expression.
//   - Replace uses with `ttg.local_load_slice` / `ttg.local_store_slice` ops,
//     preserving mask semantics.
//
// [Design]
//   - Anchors are produced by a prior pass (PromoteSeqAlignToShared).
//   - `deriveSlotOffset(ptr)` extracts the slot base via `tt.addptr(offset=slotOff)`.
//   - `deriveLaneOffsetTensor(ptr)` extracts per-lane offsets when the pointer
//     matches `addptr(splat(base), offset_tensor)`.
//   - Buffers are per-element-type, allocated once at function entry, and cached.
//   - All SMEM MemDescs use `#ttg.linear_shared` to isolate from generic transforms.
//
// [Contract]
//   - Only processes `tt.func @sw_kernel`.
//   - Requires anchors with `smem.anchor` and pointer pattern
//     `addptr(splat(base), offset_tensor)`; otherwise the pass fails with error.
//   - Emits local_*_slice ops; later lowering (Local*SliceOpConversion) will
//     perform deterministic GEP + ld/st in SMEM.
//
// [Invariants]
//   - STRIDE, BLOCK, SEGS, slot counts (H/E/F) are compile-time constants here.
//   - MemDesc ranks: {2D [SLOTS, STRIDE]} and derived 1D view for each slot.
//   - Element type of data tensors equals MemDesc element type.
//   - A GPU barrier is inserted after initialization to ensure SMEM readiness.
//
// [Diagnostics]
//   - Consistent stderr prefix: “[MaterializeSWSmem][INFO/WARN/OK/FAIL/DEBUG]”.
//   - Errors are emitted via op->emitError for hard contract violations.
//
// [Repro]
//   - Run pipeline: SeqAlignDetect → PromoteSeqAlignToShared → MaterializeSWSmem.
//   - Inspect IR for `ttg.local_{load,store}_slice` and linear_shared memdescs.
//===----------------------------------------------------------------------===//

#include <functional>
#include <limits>
#include <cstdlib>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

using namespace mlir;
namespace tt  = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

/// Compact record for an anchor discovered on a ConvertLayoutOp.
///
/// [Fields]
///   - op:          the anchor `ttg.convert_layout`
///   - originalPtr: pointer SSA value associated with the original load/store
///   - isLoad:      whether the original op is tt.load (vs tt.store)
///   - originalOp:  the source tt.load/tt.store op
///   - kindAttr:    "H" | "E" | "F"
struct AnchorInfo {
  ttg::ConvertLayoutOp op;
  Value originalPtr;
  bool isLoad;
  Operation *originalOp;
  StringAttr kindAttr;
};

/// Extract anchor information from a `ttg.convert_layout` op.
///
/// [Match]
///   - For loads:  anchor.src is defined by tt.load; ptr is load.getPtr().
///   - For stores: a user tt.store that consumes anchor.result as value; ptr is
///                 store.getPtr().
///
/// [Return]
///   std::nullopt if `smem.anchor` is missing or no matching pattern is found.
static std::optional<AnchorInfo> buildAnchorFromConvert(ttg::ConvertLayoutOp cvt) {
    auto kindAttr = cvt->getAttrOfType<StringAttr>("smem.anchor");
    if (!kindAttr) return std::nullopt;
    if (auto *def = cvt.getSrc().getDefiningOp()) {
        if (auto load = dyn_cast<tt::LoadOp>(def))
            return AnchorInfo{cvt, load.getPtr(), true, load.getOperation(), kindAttr};
    }
    for (Operation *user : cvt.getResult().getUsers()) {
        if (auto st = dyn_cast<tt::StoreOp>(user)) {
            if (st.getValue() == cvt.getResult())
                return AnchorInfo{cvt, st.getPtr(), false, st.getOperation(), kindAttr};
        }
    }
    return std::nullopt;
}

// [Design] Slot offset comes directly from the scalar base pointer's addptr.
// Old helper attempted to reconstruct more than needed; this simplified
// version matches the real IR shape: addptr(..., slotOffset).
static std::optional<Value> deriveSlotOffset(Value ptr) {
    if (auto addPtrOp = ptr.getDefiningOp<tt::AddPtrOp>()) {
        return addPtrOp.getOffset();
    }
    return std::nullopt;
}

/// Derive the per-lane logical offset tensor from a pointer expression.
///
/// [Match]
///   `addptr( splat(base), offset_tensor )`
///
/// [Return]
///   The `offset_tensor` if matched; otherwise nullopt.
static std::optional<Value> deriveLaneOffsetTensor(Value ptr) {
    if (auto addPtr = ptr.getDefiningOp<tt::AddPtrOp>()) {
        if (addPtr.getPtr().getDefiningOp<tt::SplatOp>()) {
            return addPtr.getOffset();
        }
    }
    return std::nullopt;
}


struct MaterializeSWSmem
    : public PassWrapper<MaterializeSWSmem, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeSWSmem)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tt::TritonDialect, ttg::TritonGPUDialect, arith::ArithDialect, 
                    mlir::gpu::GPUDialect>();
  }

  struct Buffers {
    Value hBuf, eBuf, fBuf;
    bool inited = false;
  };

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::errs() << "[MaterializeSWSmem][INFO] BEGIN \n";

    // [Design] Tuning constants for current SW kernel variant.
    const int64_t STRIDE = 768;
    const int64_t BLOCK  = 256;
    const int64_t HSlots = 3, ESlots = 2, FSlots = 2;

    module.walk([&](tt::FuncOp func) {
      if (func.getName() != "sw_kernel") return;

      // Collect anchor ops.
      SmallVector<ttg::ConvertLayoutOp> anchors;
      func.walk([&](ttg::ConvertLayoutOp cvt) {
        if (cvt->hasAttrOfType<StringAttr>("smem.anchor")) {
          anchors.push_back(cvt);
        }
      });

      llvm::errs() << "[MaterializeSWSmem][INFO] Found " << anchors.size() << " anchors in @" << func.getName() << "\n";
      if (anchors.empty()) return;

      // Create allocations at function entry so all users dominate them.
      Block &entry = func.getBody().front();
      OpBuilder top(&entry, entry.begin());
      auto *ctx = module.getContext();
      auto loc  = func.getLoc();

      // Encodings / address-space attrs
      // [Fix] Use flat 1D MemDescs only. MemDescType requires all dims except the
      // first to be power-of-2 (see Types.cpp::MemDescType::verify). STRIDE=768 is
      // not power-of-2, so a 2D [SLOTS, STRIDE] shape fails. A 1D [SLOTS*STRIDE]
      // shape trivially passes (drop_front(1) is empty → all_of vacuously true).
      SmallVector<unsigned> ord1{0};
      auto cta1 = ttg::CTALayoutAttr::getDefault(ctx, 1);
      auto smem = ttg::SharedMemorySpaceAttr::get(ctx);
      auto linearEnc1D = ttg::LinearSharedEncodingAttr::get(ctx, ord1, cta1);

      // Cache buffers per element type.
      llvm::DenseMap<const void*, Buffers> bufMap;
      auto getOrCreateBuffers = [&](Type elemTy) -> Buffers& {
        const void *key = elemTy.getAsOpaquePointer();
        auto it = bufMap.find(key);
        if (it != bufMap.end()) return it->second;

        Buffers B{};
        // Flat 1D allocations: [SLOTS * STRIDE] elements each.
        auto ty1D_H = ttg::MemDescType::get({HSlots * STRIDE}, elemTy, linearEnc1D, smem, true);
        auto ty1D_E = ttg::MemDescType::get({ESlots * STRIDE}, elemTy, linearEnc1D, smem, true);
        auto ty1D_F = ttg::MemDescType::get({FSlots * STRIDE}, elemTy, linearEnc1D, smem, true);

        B.hBuf = top.create<ttg::LocalAllocOp>(loc, ty1D_H).getResult();
        B.eBuf = top.create<ttg::LocalAllocOp>(loc, ty1D_E).getResult();
        B.fBuf = top.create<ttg::LocalAllocOp>(loc, ty1D_F).getResult();

        if (elemTy.isInteger(32) && !B.inited) {
            // Initialize to a large negative sentinel to preserve correctness
            // before first write (affine gap penalties).
            auto cMINF = top.create<arith::ConstantIntOp>(loc, -10000000, 32);

            // [Design] Read warp/thread config from module attributes; do not
            // rely on pseudo APIs. Fallbacks keep defaults sane.
            unsigned int numWarps = 1;
            unsigned int threadsPerWarp = 32;
            if (auto warpsAttr = module->getAttrOfType<IntegerAttr>("ttg.num-warps"))
                numWarps = warpsAttr.getInt();
            if (auto threadsAttr = module->getAttrOfType<IntegerAttr>("ttg.threads-per-warp"))
                threadsPerWarp = threadsAttr.getInt();
            unsigned int numThreads = numWarps * threadsPerWarp;

            // Initialize in BLOCK=256 chunks using a BlockedEncoding on registers.
            unsigned int sizePerThread = 2;
            auto blockedEnc256 = ttg::BlockedEncodingAttr::get(ctx, {sizePerThread}, {threadsPerWarp}, {numThreads/threadsPerWarp}, {0}, cta1);
            auto initTy256 = RankedTensorType::get({BLOCK}, elemTy, blockedEnc256);

            Value vMINF = top.create<tt::SplatOp>(loc, initTy256, cMINF);
            Value makeRange = top.create<tt::MakeRangeOp>(loc, initTy256, 0, BLOCK);
            Value c0 = top.create<arith::ConstantIntOp>(loc, 0, 32);
            Value cBlock = top.create<arith::ConstantIntOp>(loc, BLOCK, 32);

            // Strip-mine the initialization in a runtime while-loop so the
            // backend keeps a compact loop body instead of unrolling one
            // static local_store_slice per BLOCK-sized chunk.
            auto initBuffer = [&](Value buf1D, int numSlots) {
                Value limit = top.create<arith::ConstantIntOp>(loc,
                                                               numSlots * STRIDE,
                                                               32);
                auto whileOp = top.create<scf::WhileOp>(loc, TypeRange{c0.getType()},
                                                        ValueRange{c0});

                Block *before =
                    top.createBlock(&whileOp.getBefore(), whileOp.getBefore().begin(),
                                    TypeRange{c0.getType()}, SmallVector<Location>{loc});
                OpBuilder beforeBuilder = OpBuilder::atBlockEnd(before);
                Value beforeBase = before->getArgument(0);
                Value beforeCond = beforeBuilder.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::slt, beforeBase, limit);
                beforeBuilder.create<scf::ConditionOp>(loc, beforeCond,
                                                       ValueRange{beforeBase});

                Block *after =
                    top.createBlock(&whileOp.getAfter(), whileOp.getAfter().begin(),
                                    TypeRange{c0.getType()}, SmallVector<Location>{loc});
                OpBuilder bodyBuilder = OpBuilder::atBlockEnd(after);
                Value base = after->getArgument(0);
                Value splatBase = bodyBuilder.create<tt::SplatOp>(loc, initTy256, base);
                Value offsetTensor =
                    bodyBuilder.create<arith::AddIOp>(loc, makeRange, splatBase);
                Value splatLimit = bodyBuilder.create<tt::SplatOp>(loc, initTy256, limit);
                Value mask = bodyBuilder.create<arith::CmpIOp>(
                    loc, arith::CmpIPredicate::slt, offsetTensor, splatLimit);
                bodyBuilder.create<ttg::LocalStoreSliceOp>(loc, vMINF, buf1D,
                                                           offsetTensor, mask);
                Value nextBase =
                    bodyBuilder.create<arith::AddIOp>(loc, base, cBlock);
                bodyBuilder.create<scf::YieldOp>(loc, ValueRange{nextBase});
            };

            initBuffer(B.hBuf, HSlots);
            initBuffer(B.eBuf, ESlots);
            initBuffer(B.fBuf, FSlots);
            top.create<mlir::gpu::BarrierOp>(loc);
            B.inited = true;
        }

        return bufMap.try_emplace(key, B).first->second;
      };
            
      for (auto &anchor : anchors) {
        OpBuilder b(anchor);
        auto infoOpt = buildAnchorFromConvert(anchor);
        if (!infoOpt) {
          llvm::errs() << "[MaterializeSWSmem][WARN] Anchor pattern-match failed, skipping\n";
          continue;
        }
        auto info = *infoOpt;
        llvm::errs() << "[MaterializeSWSmem][OK] Anchor matched kind=" << info.kindAttr.getValue() << " isLoad=" << info.isLoad << "\n";

        auto resTensorTy = llvm::dyn_cast<RankedTensorType>(info.op.getType());
        Type elemTy = resTensorTy.getElementType();
        Buffers &B = getOrCreateBuffers(elemTy);

        // Choose the correct H/E/F flat 1D buffer and slot count.
        Value buf1D;
        int32_t slotsCount = 0;
        StringRef kind = info.kindAttr.getValue();
        if      (kind == "H") { buf1D = B.hBuf; slotsCount = HSlots; }
        else if (kind == "E") { buf1D = B.eBuf; slotsCount = ESlots; }
        else                  { buf1D = B.fBuf; slotsCount = FSlots; }

        auto locA = info.op.getLoc();

        // Recover slot/lane pointers from the original addptr pattern.
        auto addPtr = info.originalPtr.getDefiningOp<tt::AddPtrOp>();
        auto splat = addPtr.getPtr().getDefiningOp<tt::SplatOp>();
        Value scalar_base_ptr = splat.getSrc();

        // Slot index derivation from slotOffset.
        auto slotOffsetOpt = deriveSlotOffset(scalar_base_ptr);
        Value slotOffset = slotOffsetOpt.value_or(b.create<arith::ConstantIntOp>(locA, 0, 32));

        Value cStride = b.create<arith::ConstantIntOp>(locA, STRIDE, 32);
        Value cSlots  = b.create<arith::ConstantIntOp>(locA, slotsCount, 32);
        Value slotBase = b.create<arith::DivSIOp>(locA, slotOffset, cStride);
        Value slotIdx  = b.create<arith::RemSIOp>(locA, slotBase, cSlots);

        // Compute flat base offset for this slot: slotIdx * STRIDE.
        Value flatBase = b.create<arith::MulIOp>(locA, slotIdx, cStride);

        // Lane offsets tensor for this access.
        auto laneOffsetOpt = deriveLaneOffsetTensor(info.originalPtr);
        if (!laneOffsetOpt) {
            info.originalOp->emitError("MaterializeSWSmem requires addptr(splat(base), offset_tensor) pattern");
            return signalPassFailure();
        }
        Value laneOffsets = *laneOffsetOpt;

        // Build flat offset tensor: splat(slotIdx * STRIDE) + laneOffsets.
        Value splatFlatBase = b.create<tt::SplatOp>(locA, laneOffsets.getType(), flatBase);
        Value flatOffsets   = b.create<arith::AddIOp>(locA, splatFlatBase, laneOffsets);

        if (info.isLoad) {
          // Rewrite load: local_load_slice (+ optional mask select).
          Value result = b.create<ttg::LocalLoadSliceOp>(locA, resTensorTy, buf1D, flatOffsets);
          auto load = llvm::cast<tt::LoadOp>(info.originalOp);
          if (Value m = load.getMask()) {
            Value other = load.getOther();
            result = b.create<arith::SelectOp>(locA, m, result, other);
          }
          info.op.replaceAllUsesWith(result);
        } else {
          // Rewrite store: preserve the lane mask for predicated shared-store
          // lowering instead of synthesizing a read-modify-write sequence.
          auto st = llvm::cast<tt::StoreOp>(info.originalOp);
          Value toStore = info.op.getSrc();
          b.create<ttg::LocalStoreSliceOp>(locA, toStore, buf1D, flatOffsets,
                                           st.getMask());
          st.erase();
        }
        info.op.erase();
      }
    });

    llvm::errs() << "[MaterializeSWSmem][INFO] END \n";
  }
};

} // namespace

namespace mlir {
std::unique_ptr<mlir::Pass> createMaterializeSWSmem() {
  return std::make_unique<MaterializeSWSmem>();
}
} // namespace mlir
