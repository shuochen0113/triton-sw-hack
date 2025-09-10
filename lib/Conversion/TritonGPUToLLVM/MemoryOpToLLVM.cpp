#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

LogicalResult lowerLocalStore(Location loc, MLIRContext *ctx, Value regVal,
                              MemDescType memDescTy, SharedMemoryObject smemObj,
                              ArrayRef<Value> inVals,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter,
                              const TargetInfoBase &targetInfo) {
  auto regTy = cast<RankedTensorType>(regVal.getType());
  auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());

  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kOffset = str_attr("offset");
  auto regLayout = toLinearLayout(regTy);
  auto paddedLayout =
      dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(memDescTy.getEncoding());
  LinearLayout cvt = LinearLayout::empty();
  if (paddedLayout) {
    cvt = regLayout.reshapeOuts({{kOffset, regLayout.getTotalOutDimSize()}});
  } else {
    auto sharedLayout = toLinearLayout(memDescTy);
    cvt = regLayout.invertAndCompose(sharedLayout);
    auto kBlock = str_attr("block");
    // NYI. We would need to emit a map.shared::cluster instruction.
    if (!cvt.isTrivialOver({kBlock})) {
      return failure();
    }
  }
  cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
  lowerLocalLdSt(loc, ctx, cvt, inVals, llvmElemTy, memDescTy, smemObj,
                 rewriter, targetInfo);

  return success();
}

struct GlobalScratchAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::GlobalScratchAllocOp> {
  const TargetInfoBase *targetInfo;

  GlobalScratchAllocOpConversion(LLVMTypeConverter &converter,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(&targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::GlobalScratchAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto opOffsetAttr = op->getAttrOfType<mlir::IntegerAttr>(
        "ttg.global_scratch_memory_offset");
    assert(opOffsetAttr);
    auto opOffset = opOffsetAttr.getValue().getZExtValue();

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!funcOp) {
      return failure();
    }
    Value ptr = LLVM::getGlobalScratchPtr(loc, rewriter, *targetInfo, funcOp,
                                          b.i32_val(opOffset));

    rewriter.replaceOp(op, ptr);
    return success();
  }
};

struct LocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  LocalAllocOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc())
      return failure();
    Location loc = op->getLoc();
    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto memDescTy = cast<MemDescType>(op.getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, memDescTy.getRank(),
                                      loc, rewriter);
    // If there is an initial tensor, store it into the shared memory.
    if (op.getSrc()) {
      auto *ctx = op.getContext();
      auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
      if (failed(lowerLocalStore(loc, ctx, op.getSrc(), memDescTy, smemObj,
                                 inVals, typeConverter, rewriter,
                                 targetInfo))) {
        return failure();
      }
    }
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalDeallocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalDeallocOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalDeallocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::LocalDeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct LocalLoadOpConversion : public ConvertOpToLLVMPattern<LocalLoadOp> {
public:
  LocalLoadOpConversion(LLVMTypeConverter &typeConverter,
                        const TargetInfoBase &targetInfo,
                        PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescVal = op.getSrc();
    auto regVal = op.getResult();
    auto memDescTy = cast<MemDescType>(memDescVal.getType());
    auto regTy = cast<RankedTensorType>(regVal.getType());
    auto typeConverter = getTypeConverter();

    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);

    auto sharedEnc =
        cast<triton::gpu::SharedEncodingTrait>(memDescTy.getEncoding());
    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kOffset = str_attr("offset");
    auto regLayout = toLinearLayout(regTy);
    auto paddedLayout =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(sharedEnc);
    LinearLayout cvt = LinearLayout::empty();
    if (paddedLayout) {
      cvt = regLayout.reshapeOuts({{kOffset, regLayout.getTotalOutDimSize()}});
    } else {
      auto sharedLayout = toLinearLayout(memDescTy);
      cvt = regLayout.invertAndCompose(sharedLayout);
      auto kBlock = str_attr("block");
      // NYI. We would need to emit a map.shared::cluster instruction.
      if (!cvt.isTrivialOver({kBlock})) {
        return failure();
      }
    }
    cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});

    auto outVals = lowerLocalLdSt(loc, ctx, cvt, {}, llvmElemTy, memDescTy,
                                  smemObj, rewriter, targetInfo, op);

    Value result = packLLElements(loc, typeConverter, outVals, rewriter, regTy);
    rewriter.replaceOp(op, result);

    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct LocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
public:
  using ConvertOpToLLVMPattern<
      triton::gpu::LocalStoreOp>::ConvertOpToLLVMPattern;

  LocalStoreOpConversion(const LLVMTypeConverter &converter,
                         const TargetInfoBase &targetInfo,
                         PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = op.getContext();
    Value regVal = op.getSrc();
    Value memDescVal = op.getDst();
    auto typeConverter = getTypeConverter();
    auto memDescTy = cast<MemDescType>(memDescVal.getType());
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDst(),
                                                         llvmElemTy, rewriter);
    auto inVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    if (failed(lowerLocalStore(loc, ctx, regVal, memDescTy, smemObj, inVals,
                               typeConverter, rewriter, targetInfo))) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

// ============================================================================
// Lowering Pattern for ttg.local_load_slice
// [Context] Shuochen’s hack for sw_kernel v1 — 2025/09/10
//
// [Purpose]
//   Deterministic SMEM load lowering for SW kernel ring buffers. This pattern
//   consumes exactly the two artifacts preserved by the middle-end op:
//     (1) a shared-memory MemDesc tagged with #ttg.linear_shared;
//     (2) a per-thread logical offset tensor.
//   No address reconstruction or linear_tid heuristics are used.
//
// [Contract]
//   - Matches only MemDesc whose encoding is LinearSharedEncodingAttr
//     (#ttg.linear_shared). Otherwise, bail out.
//   - MemDesc is rank-1 (1D) and in #triton_gpu.shared.
//   - The logical offsets were verified at the op level to match the result
//     tensor shape (rank-1).
//
// [Invariants]
//   - `wrapBound = memDesc.shape[0]` is the 1D extent used for wrap-around.
//   - Wrap-around uses `offset & (wrapBound - 1)` which assumes wrapBound is a
//     power of two (POT) for correctness and performance.
//     NOTE: If wrapBound may be non-POT in future kernels, use `urem` instead.
//   - Element type of the result equals MemDesc element type.
//
// [Lowering Steps]
//   1) Guard on #ttg.linear_shared encoding.
//   2) Extract SMEM base address from the MemDesc struct.
//   3) Unpack the offset tensor into per-lane LLVM scalar values.
//   4) Apply wrap: `phys = offset & (wrapBound - 1)`.
//   5) Emit GEP(base, phys) + Load for each lane.
//   6) Repack loaded scalars back to the result tensor shape.
//
// [IR Before]
//   %v = ttg.local_load_slice %smem, %offset
//
// [IR After] (sketch)
//   %off_i   = extract %offset[i]
//   %mask    = const (wrapBound-1)
//   %phys_i  = and i32 %off_i, %mask
//   %ptr_i   = gep %smem_base, %phys_i
//   %val_i   = load %ptr_i
//   %result  = pack(%val_*)
//
// [Diagnostics]
//   - Failure to match encoding returns `failure()` (let other patterns try).
//   - Use small, predictable integer ops to aid SASS/PTX quality.
//
// TODO(Shuochen-2025/09/10): Add a guarded slow-path using `urem` when POT is
// not guaranteed, gated by a target/property query.
// ============================================================================

struct LocalLoadSliceOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadSliceOp> {
public:
  LocalLoadSliceOpConversion(LLVMTypeConverter &typeConverter,
                             const TargetInfoBase &targetInfo,
                             PatternBenefit benefit = 10)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();

    // --- Get types from new Op signature ---
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    auto memDescTy = cast<triton::gpu::MemDescType>(op.getSrc().getType());
    auto offsetTy = cast<RankedTensorType>(op.getOffset().getType());

    // 1. Guard: This pattern only matches our custom linear encoding.
    auto linearSharedEnc = dyn_cast<triton::gpu::LinearSharedEncodingAttr>(memDescTy.getEncoding());
    if (!linearSharedEnc)
      return failure();
      
    // --- Get shared memory info ---
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), llvmElemTy, rewriter);
    Value smemBase = smemObj.getBase();

    // The size of the 1D shared memory segment for wrap-around.
    int64_t wrapBound = memDescTy.getShape()[0];

    TritonLLVMOpBuilder b(loc, rewriter);
    SmallVector<Value> loadedVals;

    // --- Unpack the offset tensor to get per-thread logical offsets ---
    // This is the core of the new design. We get the exact offset for each element.
    auto logicalOffsets = unpackLLElements(loc, adaptor.getOffset(), rewriter);
    Value cWrapBoundMask = b.i32_val((int32_t)(wrapBound - 1));

    // For each element this thread handles...
    for (unsigned i = 0; i < logicalOffsets.size(); ++i) {
      Value logicalOffset = logicalOffsets[i];
      
      // 2. Apply wrap-around logic to the precise logical offset.
      Value finalPhysicalOffset = rewriter.create<arith::AndIOp>(loc, logicalOffset, cWrapBoundMask);
      
      // 3. Generate GEP + Load. The address calculation is now direct and simple.
      Value ptr = rewriter.create<LLVM::GEPOp>(
          loc, smemBase.getType(), llvmElemTy, smemBase, ValueRange{finalPhysicalOffset});
      Value val = rewriter.create<LLVM::LoadOp>(loc, llvmElemTy, ptr);
      loadedVals.push_back(val);
    }

    Value result = packLLElements(loc, typeConverter, loadedVals, rewriter, resultTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

// ============================================================================
// Lowering Pattern for ttg.local_store_slice
// [Context] Shuochen’s hack for sw_kernel v1 — 2025/09/10
//
// [Purpose]
//   Deterministic SMEM store lowering symmetrical to local_load_slice. Uses the
//   preserved (MemDesc, OffsetTensor) pair to emit straight GEP + store.
//
// [Contract]
//   - Matches only #ttg.linear_shared MemDesc in #triton_gpu.shared.
//   - Source data tensor and offset tensor are rank-1 and shape-equal.
//   - MemDesc is mutable and large enough for the access (op verifier ensures).
//
// [Invariants]
//   - `wrapBound = memDesc.shape[0]` is the 1D extent used for wrap-around.
//   - Wrap-around uses `offset & (wrapBound - 1)` assuming POT wrapBound.
//     NOTE: Consider a non-POT fallback with `urem` when needed.
//
// [Lowering Steps]
//   1) Guard on #ttg.linear_shared encoding.
//   2) Extract SMEM base address from the MemDesc struct.
//   3) Unpack source values and offsets into per-lane scalars.
//   4) Apply wrap: `phys = offset & (wrapBound - 1)`.
//   5) Emit GEP(base, phys) + Store for each lane.
//   6) Erase the op (no result).
//
// [IR Before]
//   ttg.local_store_slice %src, %smem, %offset
//
// [IR After] (sketch)
//   %val_i  = extract %src[i]
//   %off_i  = extract %offset[i]
//   %mask   = const (wrapBound-1)
//   %phys_i = and i32 %off_i, %mask
//   %ptr_i  = gep %smem_base, %phys_i
//   store %val_i, %ptr_i
//
// [Diagnostics]
//   - Clear separation of data vs address paths; easy to instrument for perf.
//   - Straight-line IR favors good PTX/SASS generation.
//
// TODO(Shuochen-2025/09/10): Mirror any future non-POT wrap fallback (urem) and
// add target-driven selection.
// ============================================================================

struct LocalStoreSliceOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreSliceOp> {
public:
  LocalStoreSliceOpConversion(LLVMTypeConverter &typeConverter,
                              const TargetInfoBase &targetInfo,
                              PatternBenefit benefit = 10)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();

    // --- Get types from new Op signature ---
    auto srcDataTy = cast<RankedTensorType>(op.getSrc().getType());
    auto memDescTy = cast<triton::gpu::MemDescType>(op.getDst().getType());
    auto offsetTy = cast<RankedTensorType>(op.getOffset().getType());
    
    // 1. Guard: This pattern only matches our custom linear encoding.
    auto linearSharedEnc = dyn_cast<triton::gpu::LinearSharedEncodingAttr>(memDescTy.getEncoding());
    if (!linearSharedEnc)
      return failure();
      
    // --- Get shared memory info ---
    auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDst(), llvmElemTy, rewriter);
    Value smemBase = smemObj.getBase();

    int64_t wrapBound = memDescTy.getShape()[0];
    
    TritonLLVMOpBuilder b(loc, rewriter);
    
    // --- Unpack both the data tensor and the offset tensor ---
    auto valsToStore = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto logicalOffsets = unpackLLElements(loc, adaptor.getOffset(), rewriter);
    
    Value cWrapBoundMask = b.i32_val((int32_t)(wrapBound - 1));

    for (unsigned i = 0; i < valsToStore.size(); ++i) {
      Value logicalOffset = logicalOffsets[i];

      // 2. Apply wrap-around logic.
      Value finalPhysicalOffset = rewriter.create<arith::AndIOp>(loc, logicalOffset, cWrapBoundMask);

      // 3. Generate GEP + Store.
      Value ptr = rewriter.create<LLVM::GEPOp>(
          loc, smemBase.getType(), llvmElemTy, smemBase, ValueRange{finalPhysicalOffset});
      rewriter.create<LLVM::StoreOp>(loc, valsToStore[i], ptr);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateMemoryOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<GlobalScratchAllocOpConversion>(typeConverter, targetInfo,
                                               benefit);
  patterns.add<LocalAllocOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalDeallocOpConversion>(typeConverter, benefit);
  patterns.add<LocalLoadOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalStoreOpConversion>(typeConverter, targetInfo, benefit);

  // [MODIFIED] Register our new patterns
  patterns.add<LocalLoadSliceOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<LocalStoreSliceOpConversion>(typeConverter, targetInfo, benefit);
}
