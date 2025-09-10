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
    const int64_t SEGS   = STRIDE / BLOCK;
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

      if (anchors.empty()) return;

      // Create allocations at function entry so all users dominate them.
      Block &entry = func.getBody().front();
      OpBuilder top(&entry, entry.begin());
      auto *ctx = module.getContext();
      auto loc  = func.getLoc();

      // Encodings / address-space attrs
      SmallVector<unsigned> ord2{1, 0}, ord1{0};
      auto cta2 = ttg::CTALayoutAttr::getDefault(ctx, 2);
      auto cta1 = ttg::CTALayoutAttr::getDefault(ctx, 1);
      auto smem = ttg::SharedMemorySpaceAttr::get(ctx);
      auto linearEnc2D = ttg::LinearSharedEncodingAttr::get(ctx, ord2, cta2);
      auto linearEnc1D = ttg::LinearSharedEncodingAttr::get(ctx, ord1, cta1);
      
      // Cache buffers per element type.
      llvm::DenseMap<const void*, Buffers> bufMap;
      auto getOrCreateBuffers = [&](Type elemTy) -> Buffers& {
        const void *key = elemTy.getAsOpaquePointer();
        auto it = bufMap.find(key);
        if (it != bufMap.end()) return it->second;

        Buffers B{};
        auto ty2D_H = ttg::MemDescType::get({HSlots, STRIDE}, elemTy, linearEnc2D, smem, true);
        auto ty2D_E = ttg::MemDescType::get({ESlots, STRIDE}, elemTy, linearEnc2D, smem, true);
        auto ty2D_F = ttg::MemDescType::get({FSlots, STRIDE}, elemTy, linearEnc2D, smem, true);
        
        B.hBuf = top.create<ttg::LocalAllocOp>(loc, ty2D_H).getResult();
        B.eBuf = top.create<ttg::LocalAllocOp>(loc, ty2D_E).getResult();
        B.fBuf = top.create<ttg::LocalAllocOp>(loc, ty2D_F).getResult();
        
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

            auto initBuffer = [&](Value buf2D, int numSlots) {
                auto slotTy1D = ttg::MemDescType::get({STRIDE}, elemTy, linearEnc1D, smem, true);
                for (int s = 0; s < numSlots; ++s) {
                    Value cs = top.create<arith::ConstantIntOp>(loc, s, 32);
                    Value slot1D = top.create<ttg::MemDescIndexOp>(loc, slotTy1D, buf2D, cs);
                    // Initialize segment by segment across STRIDE
                    for (int g = 0; g < SEGS; ++g) {
                        Value segOffset = top.create<arith::ConstantIntOp>(loc, g * BLOCK, 32);
                        Value splatSegOffset = top.create<tt::SplatOp>(loc, initTy256, segOffset);
                        Value makeRange = top.create<tt::MakeRangeOp>(loc, initTy256, 0, BLOCK);
                        Value offsetTensor = top.create<arith::AddIOp>(loc, splatSegOffset, makeRange);

                        top.create<ttg::LocalStoreSliceOp>(loc, vMINF, slot1D, offsetTensor);
                    }
                }
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
        if (!infoOpt) continue;
        auto info = *infoOpt;

        auto resTensorTy = llvm::dyn_cast<RankedTensorType>(info.op.getType());
        Type elemTy = resTensorTy.getElementType();
        Buffers &B = getOrCreateBuffers(elemTy);

        // Choose the correct H/E/F buffer family and slot count.
        Value buf2D;
        int32_t slotsCount = 0;
        StringRef kind = info.kindAttr.getValue();
        if      (kind == "H") { buf2D = B.hBuf; slotsCount = HSlots; }
        else if (kind == "E") { buf2D = B.eBuf; slotsCount = ESlots; }
        else                  { buf2D = B.fBuf; slotsCount = FSlots; }
        
        auto slotTy1D = ttg::MemDescType::get({STRIDE}, elemTy, linearEnc1D, smem, true);
        auto locA = info.op.getLoc();
        
        // Recover slot/lane pointers from the original addptr pattern.
        auto addPtr = info.originalPtr.getDefiningOp<tt::AddPtrOp>();
        auto splat = addPtr.getPtr().getDefiningOp<tt::SplatOp>();
        Value scalar_base_ptr = splat.getSrc();

        // [DEBUG] Print the original op and base pointer.
        // llvm::errs() << "[MaterializeSWSmem][DEBUG] Original op: ";
        // info.originalOp->print(llvm::errs());
        // llvm::errs() << "\n[MaterializeSWSmem][DEBUG] scalar_base_ptr: " << scalar_base_ptr << "\n";

        // Slot index derivation from slotOffset.
        auto slotOffsetOpt = deriveSlotOffset(scalar_base_ptr);
        Value slotOffset = slotOffsetOpt.value_or(b.create<arith::ConstantIntOp>(locA, 0, 32));
        // llvm::errs() << "[MaterializeSWSmem][DEBUG] Derived slotOffset: " << slotOffset << "\n";

        Value cStride = b.create<arith::ConstantIntOp>(locA, STRIDE, 32);
        Value cSlots = b.create<arith::ConstantIntOp>(locA, slotsCount, 32);
        Value slotBase = b.create<arith::DivSIOp>(locA, slotOffset, cStride);
        Value slotIdx = b.create<arith::RemSIOp>(locA, slotBase, cSlots);
        // llvm::errs() << "[MaterializeSWSmem][DEBUG] Calculated slotIdx: " << slotIdx << "\n";
        
        // Lane offsets tensor for this access.
        auto laneOffsetOpt = deriveLaneOffsetTensor(info.originalPtr);
        if (!laneOffsetOpt) {
            info.originalOp->emitError("MaterializeSWSmem requires addptr(splat(base), offset_tensor) pattern");
            return signalPassFailure();
        }
        Value offsetTensor = *laneOffsetOpt;

        // Index into the correct slot: [slotIdx, *]
        Value slot1D = b.create<ttg::MemDescIndexOp>(locA, slotTy1D, buf2D, slotIdx);

        if (info.isLoad) {
          // Rewrite load: local_load_slice (+ optional mask select).
          Value result = b.create<ttg::LocalLoadSliceOp>(locA, resTensorTy, slot1D, offsetTensor);
          auto load = llvm::cast<tt::LoadOp>(info.originalOp);
          if (Value m = load.getMask()) {
            Value other = load.getOther();
            result = b.create<arith::SelectOp>(locA, m, result, other);
          }
          info.op.replaceAllUsesWith(result);
        } else {
          // Rewrite store: optional masked update via read-modify-write.
          auto st = llvm::cast<tt::StoreOp>(info.originalOp);
          Value toStore = info.op.getSrc();
          
          if (Value m = st.getMask()) {
            Value oldVal = b.create<ttg::LocalLoadSliceOp>(locA, resTensorTy, slot1D, offsetTensor);
            Value newVal = b.create<arith::SelectOp>(locA, m, toStore, oldVal);
            b.create<ttg::LocalStoreSliceOp>(locA, newVal, slot1D, offsetTensor);
          } else {
            b.create<ttg::LocalStoreSliceOp>(locA, toStore, slot1D, offsetTensor);
          }
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